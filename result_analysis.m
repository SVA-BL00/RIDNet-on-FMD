% Sara Vargas Aceves 40349399
% Phuong Khanh Ton 40228962

folder_pretrained  = 'results_Standard';
folder_finetune  = 'results_Finetune';

subdatasets = {
    'Confocal_BPAE_B'
    'Confocal_BPAE_G'
    'Confocal_BPAE_R'
    'Confocal_FISH'
    'Confocal_MICE'
    'TwoPhoton_BPAE_B'
    'TwoPhoton_BPAE_G'
    'TwoPhoton_BPAE_R'
    'TwoPhoton_MICE'
    'WideField_BPAE_B'
    'WideField_BPAE_G'
    'WideField_BPAE_R'
};

n_datasets = length(subdatasets);
results_table = cell(n_datasets, 5);
results_table(1,:) = {'Subdataset', 'Finetune_PSNR', 'Pretrained_PSNR', 'Finetune_SSIM', 'Pretrained_SSIM'};

for i = 1:n_datasets
    dataset = subdatasets{i};
    pattern = [dataset, '_*'];
    files_ft = dir(fullfile(folder_finetune, [pattern, '_HR.*']));
    files_std = dir(fullfile(folder_pretrained, [pattern, '_HR.*']));
    psnr_ft = [];
    ssim_ft = [];
    psnr_std = [];
    ssim_std = [];
    

    for j = 1:length(files_ft)
        hr_name = files_ft(j).name;
        base_name = strrep(hr_name, '_HR', '');
        base_name = strrep(base_name, extractAfter(base_name, '.'), '');
        base_name = base_name(1:end-1);
        
        % FINE-TUNE
        hr_ft = imread(fullfile(folder_finetune, hr_name));
        sr_ft_name = strrep(hr_name, '_HR', '_SR');
        
        if exist(fullfile(folder_finetune, sr_ft_name), 'file')
            sr_ft = imread(fullfile(folder_finetune, sr_ft_name));
            hr_ft_d = im2double(hr_ft);
            sr_ft_d = im2double(sr_ft);
            psnr_ft(end+1) = psnr(sr_ft_d, hr_ft_d);
            ssim_ft(end+1) = ssim(sr_ft_d, hr_ft_d);
        end
        
        % pretrained
        hr_std = imread(fullfile(folder_pretrained, hr_name));
        sr_std_name = strrep(hr_name, '_HR', '_SR');
        hr_path = ''; lr_path = ''; sr_ft_path = ''; sr_std_path = '';

        if exist(fullfile(folder_pretrained, sr_std_name), 'file')
            sr_std = imread(fullfile(folder_pretrained, sr_std_name));
            hr_std_d = im2double(hr_std);
            sr_std_d = im2double(sr_std);
            psnr_std(end+1) = psnr(sr_std_d, hr_std_d);
            ssim_std(end+1) = ssim(sr_std_d, hr_std_d);
        end
    end

    results_table{i+1, 1} = dataset;
    results_table{i+1, 2} = mean(psnr_ft);
    results_table{i+1, 3} = mean(psnr_std);
    results_table{i+1, 4} = mean(ssim_ft);
    results_table{i+1, 5} = mean(ssim_std);
end

fprintf('%-25s %15s %15s %15s %15s\n', results_table{1,:});
fprintf(repmat('-', 1, 99));
fprintf('\n');
for i = 2:size(results_table, 1)
    fprintf('%-25s %15.4f %15.4f %15.4f %15.4f\n', results_table{i,1}, results_table{i,2}, results_table{i,3}, results_table{i,4}, results_table{i,5});
end

examples_to_plot = {
    'Confocal_BPAE_B_1_x50',
    'Confocal_BPAE_G_1_x50',
    'Confocal_BPAE_R_3_x50',
    'Confocal_FISH_1_x50',
    'Confocal_MICE_2_x50'
    'TwoPhoton_MICE_1_x50',
    'TwoPhoton_BPAE_B_2_x50'
    'TwoPhoton_BPAE_G_4_x50',
    'TwoPhoton_BPAE_R_2_x50'
    'WideField_BPAE_R_1_x50',
    'WideField_BPAE_B_3_x50',
    'WideField_BPAE_G_4_x50'
};

for k = 1:length(examples_to_plot)
    example_name = examples_to_plot{k};
    
    hr_path = fullfile(folder_finetune, [example_name, '_HR.png']);
    lr_path = fullfile(folder_finetune, [example_name, '_LR.png']);
    sr_ft_path = fullfile(folder_finetune, [example_name, '_SR.png']);
    sr_std_path = fullfile(folder_pretrained, [example_name, '_SR.png']);
    
    img_hr = imread(hr_path);
    img_lr = imread(lr_path);
    img_sr_ft = imread(sr_ft_path);
    img_sr_std = imread(sr_std_path);

    hr_d = im2double(img_hr);
    psnr_ft = psnr(im2double(img_sr_ft), hr_d);
    ssim_ft = ssim(im2double(img_sr_ft), hr_d);
    psnr_std = psnr(im2double(img_sr_std), hr_d);
    ssim_std = ssim(im2double(img_sr_std), hr_d);
    
    [h, w, ~] = size(img_hr);
    crop_h = round(h * 0.25);
    crop_w = round(w * 0.25);
    
    img_hr_zoom = img_hr(crop_h:end-crop_h, crop_w:end-crop_w, :);
    img_lr_zoom = img_lr(crop_h:end-crop_h, crop_w:end-crop_w, :);
    img_sr_std_zoom = img_sr_std(crop_h:end-crop_h, crop_w:end-crop_w, :);
    img_sr_ft_zoom = img_sr_ft(crop_h:end-crop_h, crop_w:end-crop_w, :);

    figure('Name', example_name);
    subplot(1,4,1);imshow(img_hr_zoom, []);title('Ground Truth');
    subplot(1,4,2);imshow(img_lr_zoom, []);title('Noisy');
    subplot(1,4,3);imshow(img_sr_std_zoom, []);title(sprintf('Pre-trained Model\nPSNR: %.2f dB, SSIM: %.4f', psnr_std, ssim_std));
    subplot(1,4,4);imshow(img_sr_ft_zoom, []);title(sprintf('Fine-tuned Model\nPSNR: %.2f dB, SSIM: %.4f', psnr_ft, ssim_ft));
end