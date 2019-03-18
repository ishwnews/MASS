MODEL=checkpoint_path_in_pre-training_stage
DATA_PATH=YOUR_DATA_PATH

CUDA_VISIBLE_DEVICES=1 python train.py \
	--exp_name unsupMT_enfr \
	--dump_path ./models/en-fr/ 						 \
	--exp_id unsupMT_enfr_bt+dae_multigpu		 		 \
	--reload_model "$MODEL,$MODEL"		 				 \
	--data_path $DATA_PATH        						 \
	--lgs 'en-fr'                                        \
	--ae_steps 'en,fr'                                   \
	--bt_steps 'en-fr-en,fr-en-fr' 					 	 \
	--word_shuffle 3                                     \
	--word_dropout 0.1                                   \
	--word_blank 0.1                                     \
	--lambda_bt '1.0'									 \
	--lambda_ae '0:1,100000:0.1,300000:0'			     \
	--encoder_only false                                 \
	--emb_dim 1024                                       \
	--n_layers 6                                         \
	--n_heads 8                                          \
	--dropout 0.1                                        \
	--attention_dropout 0.1                              \
	--gelu_activation true                               \
	--tokens_per_batch 2000                              \
	--batch_size 32                                      \
	--bptt 256                                           \
	--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
	--epoch_size 200000                                  \
	--eval_bleu true                                    \
	--stopping_criterion 'valid_en-fr_mt_bleu,10'        \
	--validation_metrics 'valid_en-fr_mt_bleu'           \
