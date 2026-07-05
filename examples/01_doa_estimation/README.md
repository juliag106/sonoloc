# 示例 01：声源定位（DOA）

仿真一个固定方位的声源，在 clean / 10 dB / 0 dB 三种噪声条件下，
分别用 SRP-PHAT 和 MUSIC 估计方位角与仰角，直观感受噪声对定位的影响。

```bash
python examples/01_doa_estimation/run.py
```

预期输出（数值随随机种子略有浮动）：

```
[ clean] true=(55,10)  srp=(50,20)  music=(50,10)
[ 10 dB] true=(55,10)  srp=(50,0)   music=(50,10)
[  0 dB] true=(55,10)  srp=(50,0)   music=(50,10)
```

网格步长默认 10°，因此误差在一个格点以内属正常；仰角在噪声下更容易抖动。
