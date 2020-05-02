train bpe
```
CUDA_VISIBLE_DEVICES=1 python main.py --exp_name bpe_test --transformer True --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs 'en,fr' --n_mono -1 --mono_dataset 'en:./sp_data/mono/all.en.bpe.pth,,;fr:./sp_data/mono/all.fr.bpe.pth,,' --para_dataset 'en-fr:,./sp_data/para/dev/newstest2013-ref.XX.bpe.pth,./sp_data/para/dev/newstest2014-fren-src.XX.bpe.pth' --mono_directions 'en,fr' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pivo_directions 'fr-en-fr,en-fr-en' --pretrained_emb './sp_data/mono/all.en-fr.bpe.vec' --pretrained_out True --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 --otf_num_processes 8 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 500000 --stopping_criterion bleu_en_fr_valid,10 --max_len=100
```

train wordpiece
```
CUDA_VISIBLE_DEVICES=3 python main.py --exp_name wordpiece_test --transformer True --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs 'en,fr' --n_mono -1 --mono_dataset 'en:./sp_data/mono/all.en.wordpiece.pth,,;fr:./sp_data/mono/all.fr.wordpiece.pth,,' --para_dataset 'en-fr:,./sp_data/para/dev/newstest2013-ref.XX.wordpiece.pth,./sp_data/para/dev/newstest2014-fren-src.XX.wordpiece.pth' --mono_directions 'en,fr' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pivo_directions 'fr-en-fr,en-fr-en' --pretrained_emb './sp_data/mono/all.en-fr.wordpiece.vec' --pretrained_out True --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 --otf_num_processes 5 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 500000 --stopping_criterion bleu_en_fr_valid,10 --max_len=100

```

reload bpe
```
CUDA_VISIBLE_DEVICES=7 python main.py --exp_name bpe_test --exp_id 49xr4vceiz --reload_model './dumped/bpe_test/49xr4vceiz/checkpoint.pth' --reload_enc 1 --reload_dec 1 --reload_dis 0 --transformer True --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs 'en,fr' --n_mono -1 --mono_dataset 'en:./sp_data/mono/all.en.bpe.pth,,;fr:./sp_data/mono/all.fr.bpe.pth,,' --para_dataset 'en-fr:,./sp_data/para/dev/newstest2013-ref.XX.bpe.pth,./sp_data/para/dev/newstest2014-fren-src.XX.bpe.pth' --mono_directions 'en,fr' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pivo_directions 'fr-en-fr,en-fr-en' --pretrained_emb './sp_data/mono/all.en-fr.bpe.vec' --pretrained_out True --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 --otf_num_processes 8 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 500000 --stopping_criterion bleu_en_fr_valid,10 --max_len=100
```

reload unigram
```
CUDA_VISIBLE_DEVICES=7 python main.py --exp_name unigram_test --exp_id kowdt1lwgx --reload_model './dumped/unigram_test/kowdt1lwgx/checkpoint.pth' --reload_enc 1 --reload_dec 1 --reload_dis 0 --transformer True --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs 'en,fr' --n_mono -1 --mono_dataset 'en:./sp_data/mono/all.en.unigram.pth,,;fr:./sp_data/mono/all.fr.unigram.pth,,' --para_dataset 'en-fr:,./sp_data/para/dev/newstest2013-ref.XX.unigram.pth,./sp_data/para/dev/newstest2014-fren-src.XX.unigram.pth' --mono_directions 'en,fr' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pivo_directions 'fr-en-fr,en-fr-en' --pretrained_emb './sp_data/mono/all.en-fr.unigram.vec' --pretrained_out True --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 --otf_num_processes 8 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 500000 --stopping_criterion bleu_en_fr_valid,10 --max_len=100
```

reload bytebpe
```
CUDA_VISIBLE_DEVICES=1 python main.py --exp_name bytebpe_test --exp_id 4aa8lgchlf --reload_model './dumped/bytebpe_test/4aa8lgchlf/checkpoint.pth' --reload_enc 1 --reload_dec 1 --reload_dis 0 --transformer True --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs 'en,fr' --n_mono -1 --mono_dataset 'en:./sp_data/mono/all.en.bytebpe.pth,,;fr:./sp_data/mono/all.fr.bytebpe.pth,,' --para_dataset 'en-fr:,./sp_data/para/dev/newstest2013-ref.XX.bytebpe.pth,./sp_data/para/dev/newstest2014-fren-src.XX.bytebpe.pth' --mono_directions 'en,fr' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pivo_directions 'fr-en-fr,en-fr-en' --pretrained_emb './sp_data/mono/all.en-fr.bytebpe.vec' --pretrained_out True --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 --otf_num_processes 8 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 500000 --stopping_criterion bleu_en_fr_valid,10 --max_len=100
```

reload wordpiece
```
CUDA_VISIBLE_DEVICES=3 python main.py --exp_name wordpiece_test --exp_id lbgamxguj1 --reload_model './dumped/wordpiece_test/lbgamxguj1/checkpoint.pth' --reload_enc 1 --reload_dec 1 --reload_dis 0 --transformer True --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs 'en,fr' --n_mono -1 --mono_dataset 'en:./sp_data/mono/all.en.wordpiece.pth,,;fr:./sp_data/mono/all.fr.wordpiece.pth,,' --para_dataset 'en-fr:,./sp_data/para/dev/newstest2013-ref.XX.wordpiece.pth,./sp_data/para/dev/newstest2014-fren-src.XX.wordpiece.pth' --mono_directions 'en,fr' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pivo_directions 'fr-en-fr,en-fr-en' --pretrained_emb './sp_data/mono/all.en-fr.wordpiece.vec' --pretrained_out True --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 --otf_num_processes 8 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 500000 --stopping_criterion bleu_en_fr_valid,10 --max_len=100
```