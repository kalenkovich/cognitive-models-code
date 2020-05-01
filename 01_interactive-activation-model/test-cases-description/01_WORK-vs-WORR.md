# Test case 1: WORK vs WORR

**The stimulus.** 

At the first three positions W, O, and R are presented and fully extracted. 
In the fourth, the features consistent with both of K and R (both present and absent) were extracted,
the features that would disambiguate the two were not available.

**Expected results.** 

1. The activations of words WORK, WORD, WEAK, and WEAK are depicted in Figure 6, top panel.
WORK quickly gets activated and inhibits WORD that only initially gets activated.
This happens because WORK is excited by all four letters while WORD - only by three.
WEAK and WEAR never get activated and after ~15 cycles plateau at around -0.15.
WEAK is a little bit less inhibited because it gets excited more strongly by a more strongly activated K.

2. The activations of letters K, R, and D in the fourth position are depicted in Figure 6, bottom panel.
K and R initially get equally activated because the extracted features are equally consisted with both of them.
Quickly, however, K gets additional reinforcement from WORK which R does not share.
There is no word-to-letter inhibition though, so R does not get deactivated but plateaus at ~0.35.
D get quickly inhibited to the minimum activation level of -0.2 by the feature-to-letter inhibitory connections.

3. Response strength for letters R, K, and D is depicted in Figure 7.
The timecourse is similar to the letter activations, just slower. R response probability is equal to K's at first,
then slows down, then approaches 0.
For D, the probability steeadily approaches 0.
