// Tiled FC1/FC2 schedule. Every role keeps its ring phases across tile
// jobs; one resident-grid barrier separates the complete FC1 and FC2 phases.
__global__ __cluster_dims__(kClusterCtas, 1, 1)
__launch_bounds__(kWarpCount * kWarpSize, 1)
void mxfp4_fc1fc2_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap sfa_map,
    const __grid_constant__ CUtensorMap sfw1_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap sfw2_map,
    const uint8_t* raw_a, const uint8_t* raw_sfa,
    const int32_t* source_rows, const ExpertTileTask* tasks,
    const int32_t* task_state, const int32_t* grouped_assignments,
    const void* routing_weights,
    __nv_bfloat16* output, uint8_t* global_workspace, float swiglu_limit,
    int tokens, int m, int topk, int row_task_count_arg) {
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    constexpr int pair_id = 0;
    const int pair_rank = cta_rank;
    const int worker_id = blockIdx.x / kClusterCtas;
    const int worker_count = gridDim.x / kClusterCtas;
    const int row_task_count = row_task_count_arg;
    auto *grid_barrier_words =
        reinterpret_cast<uint32_t *>(global_workspace + static_cast<size_t>(row_task_count_arg) *
                                                            kClusterM * kIntermediateBytes);

    __shared__ int tmem_base[1];
    __shared__ uint64_t pair_ready[kBarrierStageCount];
    __shared__ uint64_t tma_done[kBarrierStageCount];
    __shared__ uint64_t stage_free[kBarrierStageCount];
    __shared__ uint64_t gather_done[kFc1StageCount];
    __shared__ uint64_t output_done[kAccumulatorStages];
    __shared__ uint64_t epilogue_free[kAccumulatorStages];
    __shared__ uint64_t overlap_free[kAccumulatorStages];
    __shared__ uint64_t pair_gather[1];
    __shared__ uint64_t pair_release[1];
    __shared__ int task_source_rows[kTileM];
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t *intermediate_values = smem;
    // The compact epilogue slot is disjoint from the live mainloop ring.
    uint8_t *fc1_intermediate_scales = smem + kFc1ScaleStagingOffset;
    uint8_t *fc1_workspace = smem + kEpilogueSmemBytes;
    uint8_t *fc2_workspace = smem + kEpilogueSmemBytes;
    auto *finalize_smem = reinterpret_cast<__nv_bfloat16 *>(smem);
    const int fc1_workspace_addr = static_cast<int>(__cvta_generic_to_shared(fc1_workspace));
    const int fc2_workspace_addr = static_cast<int>(__cvta_generic_to_shared(fc2_workspace));
    const int pair_ready_addr = static_cast<int>(__cvta_generic_to_shared(pair_ready));
    const int tma_done_addr = static_cast<int>(__cvta_generic_to_shared(tma_done));
    const int stage_free_addr = static_cast<int>(__cvta_generic_to_shared(stage_free));
    const int gather_done_addr = static_cast<int>(__cvta_generic_to_shared(gather_done));
    const int output_done_addr = static_cast<int>(__cvta_generic_to_shared(output_done));
    const int epilogue_free_addr = static_cast<int>(__cvta_generic_to_shared(epilogue_free));
    const int overlap_free_addr = static_cast<int>(__cvta_generic_to_shared(overlap_free));
    const int pair_gather_addr = static_cast<int>(__cvta_generic_to_shared(pair_gather));
    const int pair_release_addr = static_cast<int>(__cvta_generic_to_shared(pair_release));
    if (warp == kMmaWarp) {
        const int tmem_base_addr = static_cast<int>(__cvta_generic_to_shared(tmem_base));
        asm volatile("tcgen05.alloc.cta_group::" MXFP4_FC1FC2_CTA_GROUP
                     ".sync.aligned.shared::cta.b32 "
                     "[%0], %1;" ::"r"(tmem_base_addr),
            "r"(kTmemColumns));
    }
    cluster_barrier();
    const int tmem_addr = tmem_base[0];

    if (tid == 0) {
        for (int stage = 0; stage < kBarrierStageCount; ++stage) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                             pair_ready_addr + stage * static_cast<int>(sizeof(uint64_t))),
                "r"(kReadyArrivals));
            mbarrier_init(tma_done_addr + stage * sizeof(uint64_t));
            mbarrier_init(stage_free_addr + stage * sizeof(uint64_t));
            if (stage < kFc1StageCount) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                                 gather_done_addr + stage * static_cast<int>(sizeof(uint64_t))),
                    "r"(kGatherThreadCount));
            }
        }
        for (int stage = 0; stage < kAccumulatorStages; ++stage) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                             overlap_free_addr + stage * static_cast<int>(sizeof(uint64_t))),
                "r"(kClusterM));
            mbarrier_init(output_done_addr + stage * sizeof(uint64_t));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                             epilogue_free_addr + stage * static_cast<int>(sizeof(uint64_t))),
                "r"(kClusterM));
        }
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(pair_gather_addr),
            "r"(kClusterM));
        mbarrier_init(pair_release_addr);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    cluster_barrier();
    int tma_stage = 0;
    int tma_free_phase = 1;
    int ready_tma_stage = 0;
    int ready_tma_phase = 0;
    int gather_stage = 0;
    int gather_free_phase = 1;
    int gather_done_phase = 0;
    int mma_stage = 0;
    int mma_ready_phase = 0;
    int output_phase = 0;

    int mma_accumulator_stage = 0;
    int epi_accumulator_stage = 0;
    int accumulator_free_phase = 1;
    int overlap_phase = 0;
    bool has_previous = false;

    const int fc1_task_count = row_task_count * kIntermediateAtoms;
    const int fc2_task_count = row_task_count * (kH / kTileN);
    const int fc1_jobs = (fc1_task_count > worker_id)
                             ? (fc1_task_count - worker_id + worker_count - 1) / worker_count
                             : 0;
    const int fc2_jobs = (fc2_task_count > worker_id)
                             ? (fc2_task_count - worker_id + worker_count - 1) / worker_count
                             : 0;
    for (int task = worker_id; task < fc1_task_count; task += worker_count) {
        const int row_task = task / kIntermediateAtoms;
        const int pair_i_block_begin = task % kIntermediateAtoms;
        const int pair_i_block_end = pair_i_block_begin + 1;
        const int pair_row = row_task * kClusterRows;
        ExpertTileTask row_info{pair_row, kClusterRows, 0, 0, 0};
        if constexpr (kFuseRouting) {
            row_info = tasks[row_task * kFc1TasksPerRowTask];
        }
        const int cta_tile_row = row_info.tile_row + pair_rank * kTileM;
        const int cta_row = row_info.row_start + cta_tile_row;
        const int remaining_rows = row_info.row_count - cta_tile_row;
        const int valid_rows =
            remaining_rows <= 0 ? 0 : (remaining_rows < kTileM ? remaining_rows : kTileM);
        const int expert = row_info.expert;
        if constexpr (kFuseRouting) {
            if (warp >= kGatherWarpBase && warp < kGatherWarpBase + kGatherWarpCount) {
                for (int row = (warp - kGatherWarpBase) * kWarpSize + (tid & 31); row < kTileM;
                    row += kGatherThreadCount)
                    task_source_rows[row] =
                        row < valid_rows ? __ldg(source_rows + cta_row + row) : 0;
                asm volatile("bar.sync 4, %0;" ::"r"(kGatherThreadCount) : "memory");
            }
        }
        auto *scratch = global_workspace +
                        static_cast<size_t>(row_task * kClusterM + pair_rank) * kIntermediateBytes;
        // FC1: a physical 256-column tile yields one logical 128-column block.
        if (warp == kTmaWarp && elect_sync()) {
            for (int i_block = pair_i_block_begin; i_block < pair_i_block_end; ++i_block) {
                const int physical_n = i_block * kTileN;
                for (int h_stage = 0; h_stage < kH / kStageK; ++h_stage) {
                    const int h = h_stage * kStageK;
                    const int stage_base = fc1_workspace_addr + tma_stage * kWorkspaceBytes;
                    const int stage_a = stage_base;
                    const int stage_b = stage_a + kMatrixSmemBytes;
                    const int stage_sfa = stage_b + kWeightSmemBytes;
                    const int stage_sfb = stage_sfa + kScaleStageBytes;
                    const int local_tma_done = tma_done_addr + tma_stage * sizeof(uint64_t);
                    mbarrier_wait(stage_free_addr + tma_stage * sizeof(uint64_t), tma_free_phase);
                    mbarrier_expect_tx(local_tma_done,
                        kFuseRouting ? kWeightTmaBytes + kWeightScaleTmaBytes
                                     : kMatrixSmemBytes + kWeightTmaBytes + kScaleStageBytes +
                                           kWeightScaleTmaBytes);
                    if constexpr (!kFuseRouting) {
                        tma_load_4d(stage_a, &a_map, h, cta_row, 0, 0, local_tma_done);
                    }
                    tma_load_4d(stage_b,
                        &w1_map,
                        h,
                        expert * kPhysicalFc1N + physical_n + pair_rank * (kTileN / kClusterM),
                        0,
                        0,
                        local_tma_done);
                    if constexpr (!kFuseRouting) {
                        tma_load_4d(stage_sfa,
                            &sfa_map,
                            0,
                            0,
                            h / kMmaK,
                            cta_row / kTileM,
                            local_tma_done);
                    }
                    tma_load_sfb(stage_sfb,
                        &sfw1_map,
                        h / kMmaK,
                        expert * (kPhysicalFc1N / 128) + physical_n / 128,
                        local_tma_done,
                        cta_rank);
                    tma_stage = (tma_stage + 1) % kFc1StageCount;
                    if (tma_stage == 0) {
                        tma_free_phase ^= 1;
                    }
                }
            }
        }
        if constexpr (kFuseRouting) {
            if (warp >= kGatherWarpBase && warp < kGatherWarpBase + kGatherWarpCount) {
                const int gather_rank = warp - kGatherWarpBase;
                const int lane = tid & (kWarpSize - 1);
                constexpr int kVectorsPerRow = kStageK / 16;
                constexpr int kRowsPerIssue = kWarpSize / kVectorsPerRow;
                constexpr int kRowsPerGatherWarp = kTileM / kGatherWarpCount;
                constexpr int kRowGroupCount = kRowsPerGatherWarp / kRowsPerIssue;
                int a_source_row_cache[kRowGroupCount];
                constexpr int kScaleRowsPerThread = kTileM / kGatherThreadCount;
                int sfa_source_rows[kScaleRowsPerThread];
                {
#pragma unroll
                    for (int row_group = 0; row_group < kRowGroupCount; ++row_group) {
                        const int local_row = gather_rank * kRowsPerGatherWarp +
                                              row_group * kRowsPerIssue + lane / kVectorsPerRow;
                        a_source_row_cache[row_group] = task_source_rows[local_row];
                    }
                }
                const int sfa_local_row = gather_rank * kWarpSize + lane;
                {
#pragma unroll
                    for (int row_index = 0; row_index < kScaleRowsPerThread; ++row_index)
                        sfa_source_rows[row_index] =
                            task_source_rows[sfa_local_row + row_index * kGatherThreadCount];
                }
                for (int i_block = pair_i_block_begin; i_block < pair_i_block_end; ++i_block) {
                    for (int h_stage = 0; h_stage < kH / kStageK; ++h_stage) {
                        const int h = h_stage * kStageK;
                        const int stage_base = fc1_workspace_addr + gather_stage * kWorkspaceBytes;
                        const int stage_a = stage_base;
                        const int stage_sfa = stage_base + kMatrixSmemBytes + kWeightSmemBytes;
                        mbarrier_wait(stage_free_addr + gather_stage * sizeof(uint64_t),
                            gather_free_phase);
                        const int vec = lane % kVectorsPerRow;
#pragma unroll
                        for (int row_group = 0; row_group < kRowGroupCount; ++row_group) {
                            const int local_row = gather_rank * kRowsPerGatherWarp +
                                                  row_group * kRowsPerIssue + lane / kVectorsPerRow;
                            const bool valid = local_row < valid_rows;
                            const int source_row = a_source_row_cache[row_group];
                            const size_t source_offset =
                                static_cast<size_t>(source_row) * kH + h + vec * 16;
                            const int destination = stage_a + swizzled_vector(local_row, vec) * 16;
                            cp_async_16b(destination, raw_a + source_offset, valid);
                        }
#pragma unroll
                        for (int row_index = 0; row_index < kScaleRowsPerThread; ++row_index) {
                            const int local_row = sfa_local_row + row_index * kGatherThreadCount;
                            const bool valid = local_row < valid_rows;
                            const int source_row = sfa_source_rows[row_index];
#pragma unroll
                            for (int mma_k = 0; mma_k < kMmaSteps; ++mma_k) {
                                const int source_in_tile =
                                    (source_row & 31) * 16 + ((source_row >> 5) & 3) * 4;
                                const size_t source_tile = static_cast<size_t>(source_row >> 7);
                                const size_t global_mma_k = h / kMmaK + mma_k;
                                const int destination =
                                    stage_sfa + mma_k * 512 + scale_offset(local_row, 0);
                                cp_async_4b(destination,
                                    raw_sfa + (source_tile * (kH / kMmaK) + global_mma_k) * 512 +
                                        source_in_tile,
                                    valid);
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        cp_async_mbarrier_arrive_noinc(
                            gather_done_addr + gather_stage * sizeof(uint64_t));
                        gather_stage = (gather_stage + 1) % kFc1StageCount;
                        if (gather_stage == 0) {
                            gather_free_phase ^= 1;
                            gather_done_phase ^= 1;
                        }
                    }
                }
            }
        }
        if (kClusterM > 1 && (!kReusePeerMmaWarp || pair_rank == 1) && warp == kTmaReadyWarp &&
            (kCompactWeightScales || elect_sync())) {
            for (int i_block = pair_i_block_begin; i_block < pair_i_block_end; ++i_block) {
                for (int h_stage = 0; h_stage < kH / kStageK; ++h_stage) {
                    mbarrier_wait(tma_done_addr + ready_tma_stage * sizeof(uint64_t),
                        ready_tma_phase);
                    if constexpr (kFuseRouting) {
                        mbarrier_wait(gather_done_addr + ready_tma_stage * sizeof(uint64_t),
                            gather_done_phase);
                    }
                    if constexpr (kCompactWeightScales)
                        unpack_weight_scales(fc1_workspace + ready_tma_stage * kWorkspaceBytes +
                                             kMatrixSmemBytes + kWeightSmemBytes +
                                             kScaleStageBytes);
                    if (!kCompactWeightScales || elect_sync())
                        mbarrier_arrive_on_cluster_cta(
                            pair_ready_addr + ready_tma_stage * sizeof(uint64_t),
                            pair_id * kClusterM);
                    ready_tma_stage = (ready_tma_stage + 1) % kFc1StageCount;
                    if (ready_tma_stage == 0)
                        ready_tma_phase ^= 1;
                    if constexpr (kFuseRouting) {
                        if (ready_tma_stage == 0)
                            gather_done_phase ^= 1;
                    }
                }
            }
        }
        if (pair_rank == 0 && warp == kMmaWarp && (kCompactWeightScales || elect_sync())) {
            int &accumulator_stage = mma_accumulator_stage;
            for (int i_block = pair_i_block_begin; i_block < pair_i_block_end; ++i_block) {
                mbarrier_wait(epilogue_free_addr + accumulator_stage * sizeof(uint64_t),
                    accumulator_free_phase);
                if constexpr (kAccumulatorStages == 2) {
                    if (has_previous) {
                        const int previous_stage = accumulator_stage ^ 1;
                        mbarrier_wait(overlap_free_addr + previous_stage * sizeof(uint64_t),
                            overlap_phase);
                        if (previous_stage == 1)
                            overlap_phase ^= 1;
                    }
                    has_previous = true;
                }
                const int accumulator_addr = tmem_addr + accumulator_stage * kAccumulatorStride;
                for (int h_stage = 0; h_stage < kH / kStageK; ++h_stage) {
                    const int stage_base = fc1_workspace_addr + mma_stage * kWorkspaceBytes;
                    if constexpr (kClusterM == 1 || kReusePeerMmaWarp) {
                        mbarrier_wait(tma_done_addr + mma_stage * sizeof(uint64_t),
                            mma_ready_phase);
                        if constexpr (kFuseRouting)
                            mbarrier_wait(gather_done_addr + mma_stage * sizeof(uint64_t),
                                mma_ready_phase);
                        if constexpr (kReusePeerMmaWarp)
                            mbarrier_wait(pair_ready_addr + mma_stage * sizeof(uint64_t),
                                mma_ready_phase);
                    } else {
                        mbarrier_wait(pair_ready_addr + mma_stage * sizeof(uint64_t),
                            mma_ready_phase);
                    }
                    if constexpr (kCompactWeightScales && (kClusterM == 1 || kReusePeerMmaWarp))
                        unpack_weight_scales(fc1_workspace + mma_stage * kWorkspaceBytes +
                                             kMatrixSmemBytes + kWeightSmemBytes +
                                             kScaleStageBytes);
                    if (!kCompactWeightScales || elect_sync()) {
                        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
                        for (int mma_k = 0; mma_k < kMmaSteps; ++mma_k) {
                            issue_mma_step(accumulator_addr,
                                tmem_addr + kScaleTmemCol,
                                stage_base,
                                stage_base + kMatrixSmemBytes,
                                stage_base + kMatrixSmemBytes + kWeightSmemBytes,
                                stage_base + kMatrixSmemBytes + kWeightSmemBytes + kScaleStageBytes,
                                mma_k,
                                h_stage != 0 || mma_k != 0);
                        }
                        commit_pair_mma(stage_free_addr + mma_stage * sizeof(uint64_t), pair_id);
                    }
                    mma_stage = (mma_stage + 1) % kFc1StageCount;
                    if (mma_stage == 0)
                        mma_ready_phase ^= 1;
                }
                if (!kCompactWeightScales || elect_sync())
                    commit_pair_mma(output_done_addr + accumulator_stage * sizeof(uint64_t),
                        pair_id);
                accumulator_stage = (accumulator_stage + 1) % kAccumulatorStages;
                if (accumulator_stage == 0)
                    accumulator_free_phase ^= 1;
            }
        }

        if (warp < kEpilogueWarps) {
            int &accumulator_stage = epi_accumulator_stage;
            for (int i_block = pair_i_block_begin; i_block < pair_i_block_end; ++i_block) {
                const int accumulator_addr = tmem_addr + accumulator_stage * kAccumulatorStride;
                mbarrier_wait(output_done_addr + accumulator_stage * sizeof(uint64_t),
                    output_phase);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                const int local_row = warp * kWarpSize + (tid & 31);
                const int tmem_row = pair_rank * kTileM + warp * kWarpSize;
                float activated[128];
                float abs_max = 0.0f;
                float router_scale = 1.0f;
                if constexpr (kFuseRouting && kWeightInFc1) {
                    if (local_row < valid_rows)
                        router_scale = static_cast<const float *>(
                            routing_weights)[grouped_assignments[cta_row + local_row]];
                }
                // tcgen05.ld is warp collective; whole padded warps can
                // skip it while all four warps retain the release barriers.
                if (!kFuseRouting || warp * kWarpSize < valid_rows) {
#pragma unroll
                    for (int group = 0; group < 8; ++group) {
                        float gate_values[16], up_values[16];
                        const int gate_addr = accumulator_addr + (tmem_row << 16) +
                                              (group / 4) * 128 + (group % 4) * 16;
                        const int up_addr = gate_addr + 64;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                            : "=f"(gate_values[0]),
                            "=f"(gate_values[1]),
                            "=f"(gate_values[2]),
                            "=f"(gate_values[3]),
                            "=f"(gate_values[4]),
                            "=f"(gate_values[5]),
                            "=f"(gate_values[6]),
                            "=f"(gate_values[7]),
                            "=f"(gate_values[8]),
                            "=f"(gate_values[9]),
                            "=f"(gate_values[10]),
                            "=f"(gate_values[11]),
                            "=f"(gate_values[12]),
                            "=f"(gate_values[13]),
                            "=f"(gate_values[14]),
                            "=f"(gate_values[15])
                            : "r"(gate_addr));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                            : "=f"(up_values[0]),
                            "=f"(up_values[1]),
                            "=f"(up_values[2]),
                            "=f"(up_values[3]),
                            "=f"(up_values[4]),
                            "=f"(up_values[5]),
                            "=f"(up_values[6]),
                            "=f"(up_values[7]),
                            "=f"(up_values[8]),
                            "=f"(up_values[9]),
                            "=f"(up_values[10]),
                            "=f"(up_values[11]),
                            "=f"(up_values[12]),
                            "=f"(up_values[13]),
                            "=f"(up_values[14]),
                            "=f"(up_values[15])
                            : "r"(up_addr));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
                        for (int col = 0; col < 16; ++col) {
                            float gate = gate_values[col], up = up_values[col];
                            if constexpr (kActivationBf16Rounding) {
                                gate = __bfloat162float(__float2bfloat16_rn(gate));
                                up = __bfloat162float(__float2bfloat16_rn(up));
                            }
                            if (swiglu_limit > 0.0f) {
                                gate = fminf(gate, swiglu_limit);
                                up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
                            }
                            float value = silu(gate) * up;
                            if constexpr (kWeightInFc1)
                                value *= router_scale;
                            if constexpr (kActivationBf16Rounding)
                                value = __bfloat162float(__float2bfloat16_rn(value));
                            activated[group * 16 + col] = value;
                            abs_max = fmaxf(abs_max, fabsf(value));
                        }
                    }
                }
                if constexpr (kAccumulatorStages == 2) {
                    asm volatile("bar.sync 2, 128;" ::: "memory");
                    if (warp == 0 && (tid & 31) == 0)
                        mbarrier_arrive_on_cluster_cta(
                            overlap_free_addr + accumulator_stage * sizeof(uint64_t),
                            0);
                }
                uint8_t scale_raw = 0;
                if (local_row < valid_rows) {
                    if constexpr (kActivationBf16Rounding)
                        abs_max = fmaxf(abs_max, 1.0e-4f);
                    scale_raw = encode_activation_scale(abs_max);
                    const float inverse_scale = reciprocal_e8m0(scale_raw);
                    alignas(16) uint16_t row_quantized[64];
#pragma unroll
                    for (int col = 0; col < 128; col += 2)
                        row_quantized[col / 2] =
                            __nv_cvt_float2_to_fp8x2(make_float2(activated[col] * inverse_scale,
                                                         activated[col + 1] * inverse_scale),
                                __NV_SATFINITE,
                                __NV_E4M3);
#pragma unroll
                    for (int vector = 0; vector < 8; ++vector)
                        reinterpret_cast<uint4 *>(
                            intermediate_values)[swizzled_vector(local_row, vector)] =
                            reinterpret_cast<const uint4 *>(row_quantized)[vector];
                } else {
#pragma unroll
                    for (int vector = 0; vector < 8; ++vector)
                        reinterpret_cast<uint4 *>(
                            intermediate_values)[swizzled_vector(local_row, vector)] =
                            uint4{0, 0, 0, 0};
                }
#pragma unroll
                for (int replica = 0; replica < 4; ++replica)
                    fc1_intermediate_scales[replica * 512 + scale_offset(local_row, 0)] = scale_raw;
                asm volatile("bar.sync 2, 128;" ::: "memory");
                if (warp == 0 && (tid & 31) == 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    bulk_store_bf16(
                        reinterpret_cast<__nv_bfloat16 *>(scratch + i_block * kMatrixSmemBytes),
                        reinterpret_cast<const __nv_bfloat16 *>(intermediate_values),
                        kMatrixSmemBytes);
                    bulk_store_bf16(
                        reinterpret_cast<__nv_bfloat16 *>(
                            scratch + kIntermediateValueBytes + i_block * kScaleStageBytes),
                        reinterpret_cast<const __nv_bfloat16 *>(fc1_intermediate_scales),
                        kScaleStageBytes);
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                }
                asm volatile("bar.sync 3, 128;" ::: "memory");
                if (warp == 0 && (tid & 31) == 0) {
                    mbarrier_arrive_on_cluster_cta(
                        epilogue_free_addr + accumulator_stage * sizeof(uint64_t),
                        pair_id * kClusterM);
                }
                accumulator_stage = (accumulator_stage + 1) % kAccumulatorStages;
                if (accumulator_stage == 0)
                    output_phase ^= 1;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        for (int stage = 0; stage < kBarrierStageCount; ++stage) {
            const int uses =
                (fc1_jobs * (kH / kStageK) + kBarrierStageCount - 1 - stage) / kBarrierStageCount;
            if (uses > 0)
                mbarrier_wait(stage_free_addr + stage * sizeof(uint64_t), (uses - 1) & 1);
        }
        if (cta_rank == 0) {
            for (int stage = 0; stage < kAccumulatorStages; ++stage) {
                const int uses = (fc1_jobs + kAccumulatorStages - 1 - stage) / kAccumulatorStages;
                if (uses > 0) {
                    mbarrier_wait(epilogue_free_addr + stage * sizeof(uint64_t), (uses - 1) & 1);
                    if constexpr (kAccumulatorStages == 2)
                        mbarrier_wait(overlap_free_addr + stage * sizeof(uint64_t), (uses - 1) & 1);
                }
            }
        }
    }
    __syncthreads();
    cluster_barrier();

    grid_barrier(grid_barrier_words, grid_barrier_words + 1, worker_count, 1, cta_rank, tid);
    // Drain, invalidate, and reinitialize before switching roles. PTX
    // forbids mbarrier.init on a valid object, even after completion.
    if (tid == 0) {
        for (int stage = 0; stage < kBarrierStageCount; ++stage) {
            mbarrier_invalidate(pair_ready_addr + stage * sizeof(uint64_t));
            mbarrier_invalidate(tma_done_addr + stage * sizeof(uint64_t));
            mbarrier_invalidate(stage_free_addr + stage * sizeof(uint64_t));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                             pair_ready_addr + stage * static_cast<int>(sizeof(uint64_t))),
                "r"(kReadyArrivals));
            mbarrier_init(tma_done_addr + stage * sizeof(uint64_t));
            mbarrier_init(stage_free_addr + stage * sizeof(uint64_t));
        }
        for (int stage = 0; stage < kAccumulatorStages; ++stage) {
            mbarrier_invalidate(output_done_addr + stage * sizeof(uint64_t));
            mbarrier_invalidate(epilogue_free_addr + stage * sizeof(uint64_t));
            mbarrier_invalidate(overlap_free_addr + stage * sizeof(uint64_t));
            mbarrier_init(output_done_addr + stage * sizeof(uint64_t));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                             epilogue_free_addr + stage * static_cast<int>(sizeof(uint64_t))),
                "r"(kClusterM));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                             overlap_free_addr + stage * static_cast<int>(sizeof(uint64_t))),
                "r"(kClusterM));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    cluster_barrier();

    // Reset the role-local cursors before reinterpreting the same
    // workspace as compact FC2 B/SFB stages.
    tma_stage = 0;
    tma_free_phase = 1;
    ready_tma_stage = 0;
    ready_tma_phase = 0;
    mma_stage = 0;
    mma_ready_phase = 0;
    output_phase = 0;

    mma_accumulator_stage = 0;
    epi_accumulator_stage = 0;
    accumulator_free_phase = 1;
    overlap_phase = 0;
    has_previous = false;

    for (int task = worker_id; task < fc2_task_count; task += worker_count) {
        const int row_task = task / (kH / kTileN);
        const int h_tile_begin = task % (kH / kTileN);
        const int h_tile_end = h_tile_begin + 1;
        const int pair_row = row_task * kClusterRows;
        ExpertTileTask row_info{pair_row, kClusterRows, 0, 0, 0};
        if constexpr (kFuseRouting) {
            row_info = tasks[row_task * kFc1TasksPerRowTask];
        }
        const int cta_tile_row = row_info.tile_row + pair_rank * kTileM;
        const int cta_row = row_info.row_start + cta_tile_row;
        const int remaining_rows = row_info.row_count - cta_tile_row;
        const int valid_rows =
            remaining_rows <= 0 ? 0 : (remaining_rows < kTileM ? remaining_rows : kTileM);
        const int expert = row_info.expert;
        auto *scratch = global_workspace +
                        static_cast<size_t>(row_task * kClusterM + pair_rank) * kIntermediateBytes;
        // FC2 keeps complete FP32 accumulations in TMEM while the ring
        // streams FP8 activations and MXFP4 weights from the global image.
        if (warp == kTmaWarp && elect_sync()) {
            for (int h_tile = h_tile_begin; h_tile < h_tile_end; ++h_tile) {
                const int h = h_tile * kTileN;
                for (int i_stage = 0; i_stage < kI / kStageK; ++i_stage) {
                    const int global_i = i_stage * kStageK;
                    const int stage_base = fc2_workspace_addr + tma_stage * kFc2WorkspaceBytes;
                    const int stage_a = stage_base;
                    const int stage_b = stage_a + kMatrixSmemBytes;
                    const int stage_sfa = stage_b + kWeightSmemBytes;
                    const int stage_sfb = stage_sfa + kScaleStageBytes;
                    const int local_tma_done = tma_done_addr + tma_stage * sizeof(uint64_t);
                    mbarrier_wait(stage_free_addr + tma_stage * sizeof(uint64_t), tma_free_phase);
                    // FC1's gather zero-filled padded rows in every reused
                    // A slot. Preserve those zeros and transfer only live rows;
                    // otherwise sparse experts repeatedly stream large holes.
                    const int image_value_bytes = kMatrixSmemBytes;
                    mbarrier_expect_tx(local_tma_done,
                        image_value_bytes + kWeightTmaBytes + kScaleStageBytes +
                            kWeightScaleTmaBytes);
                    if (image_value_bytes > 0)
                        bulk_load_bytes(stage_a,
                            scratch + i_stage * kMatrixSmemBytes,
                            image_value_bytes,
                            local_tma_done);
                    bulk_load_bytes(stage_sfa,
                        scratch + kIntermediateValueBytes + i_stage * kScaleStageBytes,
                        kScaleStageBytes,
                        local_tma_done);
                    tma_load_4d(stage_b,
                        &w2_map,
                        global_i,
                        expert * kH + h + pair_rank * (kTileN / kClusterM),
                        0,
                        0,
                        local_tma_done);
                    tma_load_sfb(stage_sfb,
                        &sfw2_map,
                        global_i / kMmaK,
                        expert * (kH / 128) + h / 128,
                        local_tma_done,
                        cta_rank);
                    tma_stage = (tma_stage + 1) % kFc2StageCount;
                    if (tma_stage == 0) {
                        tma_free_phase ^= 1;
                    }
                }
            }
        }
        if (kClusterM > 1 && (!kReusePeerMmaWarp || pair_rank == 1) && warp == kTmaReadyWarp &&
            (kCompactWeightScales || elect_sync())) {
            for (int h_tile = h_tile_begin; h_tile < h_tile_end; ++h_tile) {
                for (int i_stage = 0; i_stage < kI / kStageK; ++i_stage) {
                    mbarrier_wait(tma_done_addr + ready_tma_stage * sizeof(uint64_t),
                        ready_tma_phase);
                    if constexpr (kCompactWeightScales)
                        unpack_weight_scales(fc2_workspace + ready_tma_stage * kWorkspaceBytes +
                                             kMatrixSmemBytes + kWeightSmemBytes +
                                             kScaleStageBytes);
                    if (!kCompactWeightScales || elect_sync())
                        mbarrier_arrive_on_cluster_cta(
                            pair_ready_addr + ready_tma_stage * sizeof(uint64_t),
                            pair_id * kClusterM);
                    ready_tma_stage = (ready_tma_stage + 1) % kFc2StageCount;
                    if (ready_tma_stage == 0)
                        ready_tma_phase ^= 1;
                }
            }
        }
        if (pair_rank == 0 && warp == kMmaWarp && (kCompactWeightScales || elect_sync())) {
            int &accumulator_stage = mma_accumulator_stage;
            for (int h_tile = h_tile_begin; h_tile < h_tile_end; ++h_tile) {
                mbarrier_wait(epilogue_free_addr + accumulator_stage * sizeof(uint64_t),
                    accumulator_free_phase);
                if constexpr (kAccumulatorStages == 2) {
                    if (has_previous) {
                        const int previous_stage = accumulator_stage ^ 1;
                        mbarrier_wait(overlap_free_addr + previous_stage * sizeof(uint64_t),
                            overlap_phase);
                        if (previous_stage == 1)
                            overlap_phase ^= 1;
                    }
                    has_previous = true;
                }
                const int accumulator_addr = tmem_addr + accumulator_stage * kFc2AccumulatorStride;
                for (int i_stage = 0; i_stage < kI / kStageK; ++i_stage) {
                    const int stage_base = fc2_workspace_addr + mma_stage * kFc2WorkspaceBytes;
                    if constexpr (kClusterM == 1 || kReusePeerMmaWarp)
                        mbarrier_wait(tma_done_addr + mma_stage * sizeof(uint64_t),
                            mma_ready_phase);
                    if constexpr (kClusterM > 1)
                        mbarrier_wait(pair_ready_addr + mma_stage * sizeof(uint64_t),
                            mma_ready_phase);
                    if constexpr (kCompactWeightScales && (kClusterM == 1 || kReusePeerMmaWarp))
                        unpack_weight_scales(fc2_workspace + mma_stage * kWorkspaceBytes +
                                             kMatrixSmemBytes + kWeightSmemBytes +
                                             kScaleStageBytes);
                    if (!kCompactWeightScales || elect_sync()) {
                        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
                        for (int mma_k = 0; mma_k < kMmaSteps; ++mma_k) {
                            issue_mma_step(accumulator_addr,
                                tmem_addr + kFc2ScaleTmemCol,
                                stage_base,
                                stage_base + kMatrixSmemBytes,
                                stage_base + kMatrixSmemBytes + kWeightSmemBytes,
                                stage_base + kMatrixSmemBytes + kWeightSmemBytes + kScaleStageBytes,
                                mma_k,
                                i_stage != 0 || mma_k != 0);
                        }
                        commit_pair_mma(stage_free_addr + mma_stage * sizeof(uint64_t), pair_id);
                    }
                    mma_stage = (mma_stage + 1) % kFc2StageCount;
                    if (mma_stage == 0)
                        mma_ready_phase ^= 1;
                }
                if (!kCompactWeightScales || elect_sync())
                    commit_pair_mma(output_done_addr + accumulator_stage * sizeof(uint64_t),
                        pair_id);
                accumulator_stage = (accumulator_stage + 1) % kAccumulatorStages;
                if (accumulator_stage == 0)
                    accumulator_free_phase ^= 1;
            }
        }

        if (warp < kEpilogueWarps) {
            int &accumulator_stage = epi_accumulator_stage;
            for (int h_tile = h_tile_begin; h_tile < h_tile_end; ++h_tile) {
                const int h = h_tile * kTileN;
                const int accumulator_addr = tmem_addr + accumulator_stage * kFc2AccumulatorStride;
                mbarrier_wait(output_done_addr + accumulator_stage * sizeof(uint64_t),
                    output_phase);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                // A padded CTA with no routed rows has no observable TMEM
                // values. Release both slots without staging BF16 slabs.
                if constexpr (kFuseRouting) {
                    if (valid_rows == 0) {
                        asm volatile("bar.sync 2, 128;" ::: "memory");
                        if (warp == 0 && (tid & 31) == 0) {
                            if constexpr (kAccumulatorStages == 2)
                                mbarrier_arrive_on_cluster_cta(
                                    overlap_free_addr + accumulator_stage * sizeof(uint64_t),
                                    0);
                            mbarrier_arrive_on_cluster_cta(
                                epilogue_free_addr + accumulator_stage * sizeof(uint64_t),
                                0);
                        }
                        accumulator_stage = (accumulator_stage + 1) % kAccumulatorStages;
                        if (accumulator_stage == 0)
                            output_phase ^= 1;
                        continue;
                    }
                }
                const int local_row = warp * kWarpSize + (tid & 31);
                const int global_row = cta_row + local_row;
                const int tmem_row = pair_rank * kTileM + warp * kWarpSize;
                const bool valid_row = local_row < valid_rows;
                int token_row = global_row;
                float router_scale = 1.0f;
                if constexpr (kFuseRouting) {
                    if (valid_row) {
                        const int assignment = grouped_assignments[global_row];
                        token_row = assignment / topk;
                        if constexpr (!kWeightInFc1)
                            router_scale = __bfloat162float(
                                static_cast<const __nv_bfloat16 *>(routing_weights)[assignment]);
                    }
                }
                auto stage_and_store = [&](int slab, int group_begin, int group_end) {
                    for (int group = group_begin; group < group_end; ++group) {
                        float values[8];
                        const int load_addr =
                            accumulator_addr + (tmem_row << 16) + slab + group * 8;
                        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                                     "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                            : "=f"(values[0]),
                            "=f"(values[1]),
                            "=f"(values[2]),
                            "=f"(values[3]),
                            "=f"(values[4]),
                            "=f"(values[5]),
                            "=f"(values[6]),
                            "=f"(values[7])
                            : "r"(load_addr));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
                        for (int col = 0; col < 8; ++col) {
                            finalize_smem[local_row * kReduceN + group * 8 + col] =
                                __float2bfloat16_rn(values[col] * router_scale);
                        }
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (valid_row) {
                        auto *destination = output + static_cast<size_t>(token_row) * kH + h +
                                            slab + group_begin * 8;
                        auto *source = finalize_smem + local_row * kReduceN + group_begin * 8;
                        const int bytes = (group_end - group_begin) * 8 * sizeof(__nv_bfloat16);
                        if constexpr (kFuseRouting)
                            bulk_reduce_add_bf16(destination, source, bytes);
                        else
                            bulk_store_bf16(destination, source, bytes);
                    }
                    bulk_commit_and_wait_read();
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                };
                if constexpr (kAccumulatorStages == 2) {
                    const int overlap_slab = accumulator_stage == 0 ? kTileN - kReduceN : 0;
                    const int overlap_begin = accumulator_stage == 0 ? kReduceN / 8 - 2 : 0;
                    stage_and_store(overlap_slab, overlap_begin, overlap_begin + 2);
                    if (warp == 0 && (tid & 31) == 0)
                        mbarrier_arrive_on_cluster_cta(
                            overlap_free_addr + accumulator_stage * sizeof(uint64_t),
                            0);
                    if (accumulator_stage == 0)
                        stage_and_store(overlap_slab, 0, kReduceN / 8 - 2);
                    else
                        stage_and_store(overlap_slab, 2, kReduceN / 8);
#pragma unroll
                    for (int slab = 0; slab < kTileN; slab += kReduceN)
                        if (slab != overlap_slab)
                            stage_and_store(slab, 0, kReduceN / 8);
                } else {
#pragma unroll
                    for (int slab = 0; slab < kTileN; slab += kReduceN)
                        stage_and_store(slab, 0, kReduceN / 8);
                }
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                asm volatile("bar.sync 2, 128;" ::: "memory");
                if (warp == 0 && elect_sync()) {
                    mbarrier_arrive_on_cluster_cta(
                        epilogue_free_addr + accumulator_stage * sizeof(uint64_t),
                        pair_id * kClusterM);
                }
                accumulator_stage = (accumulator_stage + 1) % kAccumulatorStages;
                if (accumulator_stage == 0)
                    output_phase ^= 1;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        for (int stage = 0; stage < kBarrierStageCount; ++stage) {
            const int uses = (fc2_jobs * kIntermediateAtoms + kBarrierStageCount - 1 - stage) /
                             kBarrierStageCount;
            if (uses > 0)
                mbarrier_wait(stage_free_addr + stage * sizeof(uint64_t), (uses - 1) & 1);
        }
        if (cta_rank == 0) {
            for (int stage = 0; stage < kAccumulatorStages; ++stage) {
                const int uses = (fc2_jobs + kAccumulatorStages - 1 - stage) / kAccumulatorStages;
                if (uses > 0) {
                    mbarrier_wait(epilogue_free_addr + stage * sizeof(uint64_t), (uses - 1) & 1);
                    if constexpr (kAccumulatorStages == 2)
                        mbarrier_wait(overlap_free_addr + stage * sizeof(uint64_t), (uses - 1) & 1);
                }
            }
        }
    }
    __syncthreads();
    cluster_barrier();
    cluster_barrier();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::" MXFP4_FC1FC2_CTA_GROUP
                     ".sync.aligned.b32 %0, %1;" ::"r"(tmem_addr),
            "r"(kTmemColumns));
    }
}
