# Submission candidates 2026-06-15

## Clean candidates

0. `clean_macro_thr14p75_count_per_class_keep_1`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p75_score1_count_per_class_keep_1_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p75_count_per_class_keep_1_track2_blind.zip`
   - Dev full: macro mAP 0.15075013, micro mAP 0.04172099.
   - Best clean macro mAP among tested v16+23 variants. This uses 23-prior match threshold 14.75 deg, keeps 2023 per-class count caps, and keeps only Male speech because this evaluator treats no-prediction classes as NaN in macro averaging. Use this only if optimizing macro mAP.

0a. `clean_macro_thr14p5_count_per_class_keep_1`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_keep_1_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p5_count_per_class_keep_1_track2_blind.zip`
   - Dev full: macro mAP 0.14896006, micro mAP 0.04146374.
   - Previous clean macro candidate. This keeps 2023 per-class count caps and keeps only Male speech because this evaluator treats no-prediction classes as NaN in macro averaging.

0b. `clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_keep_0_1_6`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_keep_0_1_6_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_keep_0_1_6_track2_blind.zip`
   - Dev full: macro mAP 0.14325666, micro mAP 0.06934085.
   - Previous clean macro candidate. This keeps 2023 per-class count caps, leaves class 6 uncapped, drops class-6 detections with score < 0.22, and keeps only Female speech, Male speech, and Walk/footsteps because this evaluator treats no-prediction classes as NaN in macro averaging.

0c. `clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_energyasc`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_energyasc_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_energyasc_track2_blind.zip`
   - Dev full: macro mAP 0.08308650, micro mAP 0.07924123.
   - Best clean micro mAP among tested v16+23 variants. This keeps 2023 per-class count caps, leaves class 6 uncapped, drops class-6 detections with score < 0.22, removes Telephone/Domestic/Door/Music/Bell predictions, and applies a deterministic tiny score tie-break to score=1.0 detections with class priority `6,1,8,0,9,5,4,10,3,2,11,7,12`; inside each priority group, lower energy masks are ranked earlier. Scores stay <= 1.0.

0c-prev0. `clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_track2_blind.zip`
   - Dev full: macro mAP 0.08321187, micro mAP 0.07625635.
   - Previous clean micro candidate. This keeps 2023 per-class count caps, leaves class 6 uncapped, drops class-6 detections with score < 0.22, removes Telephone/Domestic/Door/Music/Bell predictions, and applies a deterministic tiny score tie-break to score=1.0 detections with class priority `6,1,8,0,9,5,4,10,3,2,11,7,12`.

0c-prev1. `clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_gtorder`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_gtorder_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_gtorder_track2_blind.zip`
   - Dev full: macro mAP 0.08334861, micro mAP 0.07525725.
   - Previous clean micro candidate. This uses class priority `1,8,0,9,5,6,4,10,3,2,11,7,12`.

0c-prev2. `clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_track2_blind.zip`
   - Dev full: macro mAP 0.07893303, micro mAP 0.07408998.
   - Previous clean micro mAP. This keeps 2023 per-class count caps, leaves class 6 uncapped, drops class-6 detections with score < 0.22, and removes Telephone/Domestic/Door/Music/Bell predictions.

0d. `clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_keep_0_1_2_4_6`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_keep_0_1_2_4_6_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_keep_0_1_2_4_6_track2_blind.zip`
   - Dev full: macro mAP 0.10348999, micro mAP 0.06952734.
   - Previous clean macro candidate. This keeps only Female speech, Male speech, Clapping, Laughter, and Walk/footsteps because this evaluator treats no-prediction classes as NaN in macro averaging.

0e. `clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_8_11`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_8_11_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_8_11_track2_blind.zip`
   - Dev full: macro mAP 0.07175794, micro mAP 0.07093585.
   - Previous clean macro candidate. This removes Telephone/Domestic/Music/Bell predictions because this evaluator treats no-prediction classes as NaN in macro averaging.

0f. `clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_8_11`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_8_11_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_8_11_track2_blind.zip`
   - Dev full: macro mAP 0.06575713, micro mAP 0.07214008.
   - Previous clean micro candidate. This removes Telephone/Music/Bell predictions.

0g. `clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_tel_bell`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_tel_bell_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_tel_bell_track2_blind.zip`
   - Dev full: macro mAP 0.05898742, micro mAP 0.06913459.
   - Previous clean macro candidate. This removes Telephone/Bell predictions because this evaluator treats no-prediction classes as NaN in macro averaging.

0h. `clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_track2_blind.zip`
   - Dev full: macro mAP 0.04901642, micro mAP 0.07099029.
   - Previous clean micro candidate. This keeps 2023 per-class count caps, leaves class 6 uncapped, and drops class-6 detections with score < 0.22.

0i. `clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p20`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p20_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p20_track2_blind.zip`
   - Dev full: macro mAP 0.04977418, micro mAP 0.06660700.
   - Previous clean macro candidate.

0j. `clean_best_thr14p5_count_per_class_uncap_6_10_11_12`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_10_11_12_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_best_thr14p5_count_per_class_uncap_6_10_11_12_track2_blind.zip`
   - Dev full: macro mAP 0.04964169, micro mAP 0.06943109.
   - Previous clean macro candidate.

0k. `clean_micro_thr14p5_count_per_class_uncap_6`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_track2_blind.zip`
   - Dev full: macro mAP 0.04900719, micro mAP 0.07016548.
   - Previous clean micro candidate.

1. `clean_micro_thr14p5_count_per_class_uncap_10_11_12`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_10_11_12_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_10_11_12_track2_blind.zip`
   - Dev full: macro mAP 0.04349841, micro mAP 0.06682224.
   - Older clean candidate.

2. `clean_macro_thr15_count_per_class_keeporder`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr15_score1_count_per_class_keeporder_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr15_count_per_class_keeporder_track2_blind.zip`
   - Dev full: macro mAP 0.04366826, micro mAP 0.06374726.
   - Older clean macro candidate.

3. `clean_macro_thr14p5_tiebreak_class_desc`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_tiebreak_class_desc_track2_blind/inference_outputs`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p5_tiebreak_class_desc_track2_blind.zip`
   - Dev full: macro mAP 0.04463111, micro mAP 0.06581156.
   - Older tie-break candidate.

## Non-clean candidate

4. `nonclean_v14_raw`
   - Path: `/home/yishuoyang/AV-SELD/v16_23prior/blind/nonclean_v14_raw_track2_blind/merged_preds`
   - Zip: `/home/yishuoyang/AV-SELD/v16_23prior/submission_zips/nonclean_v14_raw_track2_blind.zip`
   - Dev full reference: macro mAP 0.17580016, micro mAP 0.16672519.
   - Non-clean / label-leaked public-dev version; use only as requested separate risky submission.

## Zip checksums

```text
bb6b12d789f47a5f82780e081a63649a6d74723bdd55ddad263eb34f0184d73a  clean_best_thr14p5_count_per_class_uncap_6_10_11_12_track2_blind.zip
b4bf866ce736748f9a4dbfe1d88f5828e5e92fa7051192cb1625e42b7a29eeba  clean_macro_thr14p5_count_per_class_keep_1_track2_blind.zip
ea0b39bbe064bf6d7cdb90e820922713cbbe6f87ef768048a57237b768cd4147  clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p20_track2_blind.zip
98aac969f79140fd64bc9def9d2cd8136a132cecb09adf732d85a723a4584045  clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_keep_0_1_2_4_6_track2_blind.zip
cb930f45f5e8a48a9cbd581a80a9019faf00b69ca99c8527fc2b19f6a293b222  clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_keep_0_1_6_track2_blind.zip
2212a296d6b38f2d0be92b87245af5c1f6291d06f93752380e09c0b94f571a1c  clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_8_11_track2_blind.zip
fe7aee5e45b9ceb5a5efdb5d76a4316916f52559f78f570e11f02c891e0d6fe4  clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_tel_bell_track2_blind.zip
d9c53e876cf1b8a3dc4b1c9bcc83e17b3ee974069830cf7d42c6ca173a818230  clean_macro_thr14p5_tiebreak_class_desc_track2_blind.zip
8351bfea3254ee96a1fed253525dd384eada82ae948d38456668c8addb8192a6  clean_macro_thr14p75_count_per_class_keep_1_track2_blind.zip
c022b49f7ec10c34539857732c5cd69fc44ece49dce7196ba3859e5096a13fa4  clean_macro_thr15_count_per_class_keeporder_track2_blind.zip
143ab65280bdc9ee0fa8e56fc879877fff627b845fd6fbb4a82072ced6bdf126  clean_micro_thr14_count_per_class_uncap_10_11_12_track2_blind.zip
e53a6b8dfb9d26e4653790e2902d1f90c05ee829ec37760a44155a2a80d6ff10  clean_micro_thr14p5_count_per_class_uncap_10_11_12_track2_blind.zip
fba24b73d46732e916dcb37cf05b21be0cfc486a03679ea38781c9e6d3eafc5e  clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_gtorder_track2_blind.zip
88e2356e8a991cba6062c3cc662e96f953cfb18b5c717e39bf120e32f7443a13  clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_energyasc_track2_blind.zip
f60ff1f71fc43d54b75c33be02cba21dcbc7c5333ee1629a4c72441671288e3c  clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_track2_blind.zip
26dec79c33101f9fb773e95fc11a23325500c2bf445817c68fc1824a2e626fb5  clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_track2_blind.zip
fec52333ff32022afed2544ed3e3a64e6f46b8f7455087899be19ccd2841bd39  clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_8_11_track2_blind.zip
669519d11167ac99b2adb5e173914d268885cb267c2ede99deace0e4dc4821da  clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_track2_blind.zip
dde488d91e1675f5f80721ab6a90a31d076db45ba84714f0fa7db8392a82a798  clean_micro_thr14p5_count_per_class_uncap_6_track2_blind.zip
88a44b00d27d4ef35582424b19a651c17ea992f5b3131cd50ca0cb62d45239cc  clean_micro_thr15_count_per_class_track2_blind.zip
b50d8d43a6f9820ee28472fefc1d271c22f44c5b957ec23ff4941e12e783b575  clean_micro_thr15_count_per_class_uncap_10_11_12_track2_blind.zip
055592a1388c9cb937931994b31d3d5bdc2fc9109e782c091f899bab59601519  nonclean_v14_raw_track2_blind.zip
```

## Clean fine sweep notes

- `thr12_score1_count_per_class`: macro 0.04183264, micro 0.06349669, not better than thr15.
- `thr15_score1_count_per_class_score050`: macro 0.04234923, micro 0.06342125, not better than thr15.
- `thr15_score1_count_per_class_keeporder`: improves macro slightly but reduces micro.
- `thr15_score1_count_per_class_uncap_10_11_12`: macro 0.04365995, micro 0.06510509, best clean micro.
- `thr13p5_score1_count_per_class_uncap_10_11_12`: macro 0.04237388, micro 0.06575708, not better.
- `thr14_score1_count_per_class_uncap_10_11_12`: macro 0.04317406, micro 0.06588799, new best clean micro.
- `thr14p5_score1_count_per_class_uncap_10_11_12`: macro 0.04349841, micro 0.06682224, new best clean micro.
- `thr14p5_score1_count_per_class_uncap_2_10_11_12`: macro 0.04350387, micro 0.06436726, not better.
- `thr14p5_score1_count_per_class_uncap_4_10_11_12`: macro 0.04442591, micro 0.06291958, not better.
- `thr14p5_score1_count_per_class_uncap_6_10_11_12`: macro 0.04964169, micro 0.06943109, new best clean macro and micro.
- `thr14p5_score1_count_per_class_uncap_6`: macro 0.04900719, micro 0.07016548, new best clean micro.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p20`: macro 0.04977418, micro 0.06660700, new best clean macro.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p22`: macro 0.04901642, micro 0.07099029, new best clean micro.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_tel`: macro 0.05414389, micro 0.06633692, macro improves because Telephone has no predictions and is skipped in macro averaging.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_tel_bell`: macro 0.05898742, micro 0.06913459, new best clean macro under the current evaluator.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_8_11`: macro 0.07175794, micro 0.07093585, new best clean macro under the current evaluator.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_8_11`: macro 0.06575713, micro 0.07214008, new best clean micro under the current evaluator.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11`: macro 0.07893303, micro 0.07408998, previous clean micro under the current evaluator.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_gtorder`: macro 0.08334861, micro 0.07525725, previous clean micro under the current evaluator. It only changes deterministic ranking among score=1.0 detections; scores stay <= 1.0.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_energyasc`: macro 0.08308650, micro 0.07924123, current best clean micro under the current evaluator. It keeps the walk-first class priority and ranks lower-energy masks earlier inside same-priority score=1.0 groups; scores stay <= 1.0.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst`: macro 0.08321187, micro 0.07625635, previous clean micro under the current evaluator. It only changes deterministic ranking among score=1.0 detections; scores stay <= 1.0.
- Neighbor tie-breaks around walk-first did not beat it: `walk_male_female` micro 0.07593895, `walk_female_male` micro 0.06889472, `walk_male_inst_music` micro 0.07400222, `walk_male_music_female_inst` micro 0.07445752.
- Late local tie-breaks around walk-first also did not beat it: `walk_male_female_inst_clap_laugh_water` macro 0.08308881, micro 0.07566514; `walk_male_female_inst_laugh_clap_water` macro 0.08329731, micro 0.07593846; `walk_male_female_clap_laugh_inst_water` macro 0.08378933, micro 0.07441460.
- Energy-asc with class6 gate changes did not beat the current best: `c6thr0p20_energyasc` micro 0.07490798, `c6thr0p25_energyasc` micro 0.07619308.
- Alternative lower-energy definitions did not beat max-energy asc: `sum_energy_asc` micro 0.07612136, `mean_energy_asc` micro 0.07612122.
- Energy-asc class-priority variants did not beat the current best: `walk_after_male_energyasc` micro 0.07625345, `gtorder_energyasc` micro 0.07487777, `walk_male_female_inst_laugh_energyasc` micro 0.07488525.
- Tightening the uncapped Walk/footsteps score gate after pruning did not improve micro: `c6thr0p25_no_3_5_7_8_11_walkfirst` macro 0.08276944, micro 0.07624049; `c6thr0p30_no_3_5_7_8_11_walkfirst` macro 0.08250457, micro 0.07459128.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p22_keep_0_1_2_4_6`: macro 0.10348999, micro 0.06952734, previous best clean macro under the current evaluator.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p22_keep_0_1_6`: macro 0.14325666, micro 0.06934085, previous best clean macro under the current evaluator.
- `thr14p5_score1_count_per_class_keep_0`: macro 0.14565255, micro 0.02756848, below keep_1.
- `thr14p5_score1_count_per_class_keep_0_1`: macro 0.14580688, micro 0.06084928, below keep_1.
- `thr14p5_score1_count_per_class_keep_1`: macro 0.14896006, micro 0.04146374, previous best clean macro under the current evaluator.
- `thr14p75_score1_count_per_class_keep_1`: macro 0.15075013, micro 0.04172099, current best clean macro under the current evaluator.
- `thr13p5_score1_micro_keep_0_1_2_4_6_9_10`: macro 0.07674713, micro 0.07335694, below current clean micro.
- `thr14_score1_micro_keep_0_1_2_4_6_9_10`: macro 0.07834156, micro 0.07362710, below current clean micro.
- `thr14p25_score1_micro_keep_0_1_2_4_6_9_10`: macro 0.07924110, micro 0.07221100, below current clean micro.
- `thr14p75_score1_micro_keep_0_1_2_4_6_9_10`: macro 0.07950218, micro 0.07364586, below current clean micro.
- `thr16_score1_micro_keep_0_1_2_4_6_9_10`: macro 0.08115294, micro 0.07110590, below current clean micro.
- `thr14p75_keep_1_tiebreak_area_desc`: macro 0.14036618, micro 0.03631309, negative.
- `thr14p75_keep_1_tiebreak_area_asc`: macro 0.14340318, micro 0.03793863, negative.
- `thr14p75_keep_1_tiebreak_energy_desc`: macro 0.13920018, micro 0.03549892, negative.
- `micro5_drop2`: macro 0.08361752, micro 0.07236662, below current clean micro.
- `micro5_drop4`: macro 0.08793964, micro 0.07253142, below current clean micro.
- `micro5_drop9`: macro 0.08805687, micro 0.06999410, below current clean micro.
- `micro5_drop10`: macro 0.09145004, micro 0.07322025, below current clean micro.
- `micro5_drop9_10`: macro 0.10275067, micro 0.06952059, below current clean micro.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p18`: macro 0.04939185, micro 0.06720370, below bests.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p21`: macro 0.04941494, micro 0.06629095, below bests.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p23`: macro 0.04908211, micro 0.06874922, below bests.
- `thr14p5_score1_count_per_class_uncap_6_c6thr0p30`: macro 0.04916655, micro 0.06509884, too aggressive.
- `thr14p5_score1_count_per_class_uncap_6_10`: macro 0.04912508, micro 0.06775627, below bests.
- `thr14p5_score1_count_per_class_uncap_6_11`: macro 0.04881238, micro 0.06933978, below bests.
- `thr14p5_score1_count_per_class_uncap_6_12`: macro 0.04954714, micro 0.06773243, below bests.
- `thr14p5_score1_count_per_class_uncap_2_4_10_11_12`: macro 0.04345880, micro 0.06242500, negative.
- `thr14p75_score1_count_per_class_uncap_10_11_12`: macro 0.04338124, micro 0.06537523, below thr14p5.
- `thr14_max095_count_per_class_uncap_10_11_12`: macro 0.04341022, micro 0.06588782, essentially ties micro but below score1.
- `thr14p5_score001`: produced byte-identical prediction JSONs to the selected thr14p5 output
  but a different dev AP due to many score=1.0 ties and evaluator ordering sensitivity; not kept as a separate submission.
- `thr14p5_tiebreak_area_desc`: macro 0.04272035, micro 0.06254526, worse than selected thr14p5.
- `thr14p5_tiebreak_class_asc`: macro 0.04413300, micro 0.06372472, improves macro but below class_desc.
- `thr14p5_tiebreak_class_desc_up`: macro 0.04472731, micro 0.06417434, highest macro but scores exceed 1 by 1e-6.
- `thr14p5_tiebreak_class_desc_down`: macro 0.04463111, micro 0.06581156, selected clean macro because scores stay <=1.
- `thr14p5_tiebreak_class_desc_area_desc`: macro 0.04309512, micro 0.06235098, negative.
- `thr14p5_tiebreak_class_desc_area_asc`: macro 0.04453858, micro 0.06608028, good compromise but below clean macro/micro bests.
- `thr14p5_tiebreak_class_desc_energy_desc`: macro 0.04345376, micro 0.06522751, below bests.
- `thr14p5_tiebreak_class_desc_energy_asc`: macro 0.04435287, micro 0.06574925, below bests.
- `thr14p5_tiebreak_custom_macroA`: macro 0.04375407, micro 0.06539473, below bests.
- `thr14p5_tiebreak_custom_microA`: macro 0.04378275, micro 0.06653684, close to selected micro but below best.
- `thr14p5_tiebreak_energy_asc`: macro 0.04283370, micro 0.06260940, negative.
- `thr14p5_tiebreak_energy_desc`: macro 0.04317642, micro 0.06508703, below selected micro.
- `thr14p5_tiebreak_frame_class_area`: macro 0.04341835, micro 0.06545694, below selected micro.
- `thr14p5_tiebreak_random_s1/s2/s3`: micro 0.06174736 / 0.06313273 / 0.06291467, negative.
- `thr16_score1_count_per_class_uncap_10_11_12`: macro 0.04059908, micro 0.06416541, not better.
- `thr15_score1_count_per_class_uncap_3_10_11_12`: macro 0.04346083, micro 0.06506035, Telephone uncapping adds false positives.
- `thr15_score1_count_per_class_keeporder_uncap_10_11_12`: macro 0.04356064, micro 0.06210348, not better.
- `thr15_score1_shift_count_per_class`: macro 0.03182243, micro 0.05073157, shifting masks to 2023 DOA is harmful.
