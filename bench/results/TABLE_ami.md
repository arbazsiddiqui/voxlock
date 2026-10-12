
### AMI real meetings, headset wearer = caller (134 windows, never used for training)

| Model | Access | Real-time | WER % | Caller words missed % | Deleted % | Inserted % | Leaked words / min of background speech | WER vs VoxLock | Missed vs VoxLock | Leak vs VoxLock |
|---|---|---|---|---|---|---|---|---|---|---|
| **VoxLock** | open | yes | 10.12 | 6.77 | 2.21 | 3.35 | 2.12 | - | - | - |
| ai-coustics Quail Voice Focus 2.2 S | paid | yes | 11.85 | 8.67 | 3.73 | 3.18 | 1.06 | +1.74 [+0.27, +4.09] * | +1.91 [+0.48, +4.22] * | -1.03 [-1.87, -0.13] ! |
| ai-coustics Quail Voice Focus 2.2 L | paid | yes | 12.13 | 8.88 | 3.90 | 3.25 | 1.72 | +1.98 [+0.10, +4.75] * | +2.08 [+0.35, +4.85] * | -0.38 [-1.33, +0.65] |
| Sanas Voice Isolation SE 1.0 | paid | yes | 13.30 | 9.09 | 2.94 | 4.22 | 5.30 | +3.24 [+1.51, +5.75] * | +2.36 [+1.06, +4.10] * | +3.27 [-0.74, +10.79] |
| TIGER + caller pick | open | no | 22.84 | 9.74 | 4.66 | 13.10 | 36.85 | +12.71 [+9.48, +15.88] * | +2.91 [+0.58, +5.40] * | +34.72 [+24.69, +45.55] * |
| DeepFilterNet3 (noise only) | open | yes | 26.78 | 13.10 | 7.88 | 13.68 | 38.84 | +16.77 [+11.38, +23.24] * | +6.32 [+2.87, +10.64] * | +37.31 [+18.21, +63.79] * |
| Hush (weya-ai) | open | yes | 33.00 | 11.89 | 5.94 | 21.11 | 68.93 | +22.94 [+18.31, +27.82] * | +5.17 [+2.19, +8.81] * | +66.91 [+52.17, +81.76] * |
| No processing | - | - | 33.03 | 8.05 | 3.25 | 24.98 | 78.34 | +23.12 [+17.94, +29.39] * | +1.42 [-0.24, +4.25] | +76.73 [+59.98, +99.37] * |

`*` VoxLock better with the 95% interval excluding zero; `!` VoxLock worse with the interval excluding zero; intervals are a speaker-clustered bootstrap over files.
