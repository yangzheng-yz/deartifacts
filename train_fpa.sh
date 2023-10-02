python main.py \
--mode train \
--dataset_folder /mnt/samsung/zheng_data/datasets/NIRI_to_NIRII \
--dataset_name NIRI_to_NIRII \
--FPA_mask_source /mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/et_2-500ms-250_550-1000ms-10_1500-5000ms-8_1800LP_contrast.raw \
--results_folder /mnt/samsung/zheng_data/training_log/FPADDM_l1predimg \
--timesteps 50 \ 
--train_batch_size 2 \
--total_epoch 200 \
--start_save_epoch 1 \
--save_and_sample_every 5 \
--load_epoch 0 \
--deart_batch_size 1 \
--device_num 0 \
--loss_type l1_pred_img


