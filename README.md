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
There wasn't much difference between the two versions above, suggesting that the small variations in attention was not changing student behavior in patterned ways. Thus, I scaled up attention variance to `rand(0, 2)` for versions 1.2 and 1.3 and we see a bigger visual differnce (though not much in average loss).

* Scale up to rand(0,2) is important because it centers average attention to 1. This means that on average varying attention is still giving us the same ability, hence we can compare a model that knows the true attention versus one that just assumes constant attention = 1




V1.71, 1.72
- there seems to be severe overfitting -- the model performs better than if the model knows the actual attentions, by using the item responses and skewing student's real ability toward the probabilistic draw of what actually happened.
- some overfitting occurs even when we constrain the model to make approx-0.1-step-size random walks
  - I don't know how to actually force the model to make a categorical, noncontinuous decision among [-0.1, 0, 0.1] each step

V1.73
- bound the model prediction in the range [0, 2]: achieves much better predicted attentions and slightly better abilities, but on the whole, abilities are still significantly being overpredicted and attentions being underpredicted (avg value < 1).

---

Looking at actual student attentions as measured by unlabeled EKG data, we can see a few categorical trends. Some students have mostly constant attention, some start higher than average, drops a bit, then ends high, and some starts lower than average, increases, then ends low.

We can try to incorporate this categorical behavior into our simulation.
