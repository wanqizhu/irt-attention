V1:
- difficulty + attention -> ability
- attention: linear interpolation during the exam rand(0, 1) -- rand(0, 1)
- ability & difficulty: N(0, 1)
- guess 0.25
- sigmoid(attetion * ability - difficulty)

V1.1:
- same setting as V1
- attention hidden from model (set it to 1 so it's as if model doesn't know attention exists)

Randomness is all seeded for reproducibility.
There wasn't much difference between the two versions above, suggesting that the small variations in attention was not changing student behavior in patterned ways. Thus, I scaled up attention variance to `rand(0, 3)` for versions 1.2 and 1.3 and we see a bigger visual differnce (though not much in average loss).