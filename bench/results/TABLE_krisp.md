
### Krisp Voice Isolation Benchmark (265 real recordings, second voice in the room; Parakeet TDT 0.6B v2, Krisp's scoring)

| Model | Access | Real-time | WER % | Caller words missed % | Deleted % | Inserted % | Leaked words / min of background speech | WER vs VoxLock | Missed vs VoxLock | Leak vs VoxLock |
|---|---|---|---|---|---|---|---|---|---|---|
| **VoxLock** | open | yes | 4.95 | 4.22 | 0.92 | 0.73 | 0.11 | - | - | - |
| Krisp VI 2.5 Default | paid | yes | 5.30 | 4.21 | 0.89 | 1.09 | 1.52 | +0.34 [-0.19, +0.94] | -0.01 [-0.28, +0.29] | +1.40 [-0.05, +3.73] |
| Krisp VI 2.5 Balanced (264 files) | paid | yes | 5.32 | 4.06 | 0.91 | 1.26 | 2.02 | +0.35 [-0.17, +0.92] | -0.17 [-0.49, +0.17] | +1.90 [+0.12, +4.57] * |
| Krisp VI 2.5 HD | paid | yes | 5.79 | 4.37 | 0.95 | 1.43 | 2.74 | +0.84 [+0.02, +1.84] * | +0.15 [-0.20, +0.52] | +2.59 [+0.43, +5.41] * |
| Krisp VI 2.5 Lite | paid | yes | 6.21 | 4.26 | 0.95 | 1.94 | 4.52 | +1.22 [+0.33, +2.07] * | +0.05 [-0.22, +0.33] | +4.25 [+0.95, +7.73] * |
| ai-coustics Quail Voice Focus 2.2 L | paid | yes | 6.84 | 6.03 | 2.46 | 0.81 | 0.66 | +1.92 [+1.00, +3.08] * | +1.84 [+0.93, +3.03] * | +0.56 [-0.14, +1.66] |
| ai-coustics Quail Voice Focus 2.2 S | paid | yes | 6.91 | 6.01 | 2.56 | 0.90 | 1.00 | +1.99 [+1.15, +3.04] * | +1.81 [+1.02, +2.78] * | +0.91 [+0.12, +2.01] * |
| REAL-TSE offline (same auto sample) | open | no | 7.55 | 4.17 | 0.93 | 3.38 | 10.19 | +2.68 [+1.09, +4.87] * | -0.04 [-0.54, +0.64] | +10.29 [+5.13, +17.36] * |
| Sanas Voice Isolation SE 1.0 | paid | yes | 9.38 | 4.90 | 1.31 | 4.48 | 13.76 | +4.35 [+2.86, +5.68] * | +0.67 [+0.23, +1.24] * | +13.41 [+7.68, +18.94] * |
| REAL-TSE causal, VoxLock's starting checkpoint (same auto sample) | open | yes | 9.81 | 5.84 | 1.73 | 3.97 | 13.73 | +4.94 [+2.95, +7.36] * | +1.62 [+0.69, +2.71] * | +13.89 [+7.84, +22.33] * |
| Hush (weya-ai) | open | yes | 12.09 | 6.27 | 1.88 | 5.82 | 19.58 | +7.27 [+5.44, +9.84] * | +2.11 [+1.23, +3.23] * | +19.72 [+14.29, +26.06] * |
| DeepFilterNet3 (noise only) | open | yes | 13.30 | 6.46 | 2.60 | 6.84 | 22.77 | +8.33 [+6.08, +10.52] * | +2.22 [+1.03, +3.76] * | +22.67 [+15.42, +29.99] * |
| TIGER + caller pick | open | no | 14.72 | 3.96 | 0.75 | 10.76 | 37.24 | +9.80 [+7.80, +11.79] * | -0.26 [-0.58, +0.08] | +37.21 [+31.81, +42.12] * |
| No processing | - | - | 23.21 | 4.28 | 1.01 | 18.93 | 66.48 | +18.17 [+15.61, +20.76] * | +0.08 [-0.34, +0.57] | +66.02 [+58.54, +73.65] * |
| SAM-Audio small (Meta), text prompt + 3 s anchor | open | no | 33.77 | 27.98 | 24.40 | 5.80 | 21.15 | +28.99 [+22.87, +35.24] * | +23.84 [+18.36, +29.29] * | +21.34 [+15.32, +28.84] * |

`*` VoxLock better with the 95% interval excluding zero; `!` VoxLock worse with the interval excluding zero; intervals are a speaker-clustered bootstrap over files.
