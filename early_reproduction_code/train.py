import gc

# from preprocess import *


from helpers import *
from settings import *

# get logger
logger = LOGGER

# check if GPU is available
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")


def train(dataset_path: str, dataset_name: str, train_patient_num: int, test_patient_num: int):
    with torch.autograd.set_detect_anomaly(True):
        result_dict_test = None

        # fetch the hdf5 file
        # hdf5_file = h5py.File(os.path.join(dataset_path, 'BRATS2018_Cropped_Normalized_preprocessed_HGG' + '.h5'),
        #                       mode='r')
        # group = hdf5_file.require_group("preprocessed")
        # num_patients = group["training_data"].shape[0]
        # print(group['training_data'][0])
        # first_patient = group['training_data'][0]
        # hdf5_file.close()

        n_dataloder, dataloader_for_viz \
            = create_dataloaders(h5file_dir=dataset_path, parent_name="preprocessed", dataset_type=dataset_name,
                                 dataset_name="training_data", load_seg=False,
                                 transform_fn=[
                                     Resize(size=(SPATIAL_SIZE_FOR_TRAINING[0], SPATIAL_SIZE_FOR_TRAINING[1])),
                                     ToTensor()],
                                 apply_normalization=True, which_normalization=None,
                                 resize_slices=RESIZE_SLICES, get_viz_dataloader=True, num_workers=N_CPUS,
                                 load_indices=None, dataset='BRATS2018', shuffle=False, pin_memory=PIN_MEMORY)

        test_patient = []
        for k in range(0, test_patient_num):
            test_patient.append(dataloader_for_viz.getitem_via_index(train_patient_num + k))  # tehre should be no +1

        # loss functions
        criterion_GAN = torch.nn.MSELoss()  # BCELoss()
        criterion_pixelwise = torch.nn.L1Loss()
        mse_fake_vs_real = torch.nn.MSELoss()

        # Calculate output of image discriminator (PatchGAN)
        patch = (4, SPATIAL_SIZE_FOR_TRAINING[0] // 2 ** 4, SPATIAL_SIZE_FOR_TRAINING[1] // 2 ** 4)

        # initialize generator and discriminator
        # not sure to use tanh/relu/or none
        generator = GeneratorUNet(in_channels=4, out_channels=4, with_relu=True, with_tanh=False)
        discriminator = Discriminator(in_channels=4, dataset='BRATS2018')

        # to save results
        results_path = os.path.join(TOP_LEVEL_PATH, 'out', 'scenario_results_%s' % dataset_name)
        if not os.path.isdir(results_path):
            logger.info("make dir for scenario_results: %s" % results_path)
            os.mkdir(results_path)

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=ADAM_LEARNING_RATE, betas=(ADAM_BETA_1, ADAM_BETA_2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=ADAM_LEARNING_RATE,
                                       betas=(ADAM_BETA_1, ADAM_BETA_2))

        # if cuda is available, send everything to GPU
        if cuda:
            generator = nn.DataParallel(generator.cuda())
            discriminator = nn.DataParallel(discriminator.cuda())
            criterion_GAN.cuda()
            criterion_pixelwise.cuda()
            mse_fake_vs_real.cuda()
        else:
            generator = nn.DataParallel(generator)
            discriminator = nn.DataParallel(discriminator)

        # init networks and optimizers
        # Initialize weights
        if START_EPOCH != 0:
            # Load pretrained models
            logger.info('Loading previous checkpoint!')
            generator, optimizer_G \
                = load_checkpoint(generator, optimizer_G,
                                  os.path.join(TOP_LEVEL_PATH, 'out', dataset_name,
                                               "{}_param_{}_{}.pkl".format('generator', dataset_name, START_EPOCH)),
                                  pickle_module=pickle, device=device)
            discriminator, optimizer_D \
                = load_checkpoint(discriminator, optimizer_D,
                                  os.path.join(TOP_LEVEL_PATH, 'out', dataset_name,
                                               "{}_param_{}_{}.pkl".format('discriminator', dataset_name, START_EPOCH)),
                                  pickle_module=pickle, device=device)
        else:
            generator.apply(weights_init_normal)
            discriminator.apply(weights_init_normal)

        # define tensor type
        tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # start training

        # Book keeping
        train_hist = {
            'D_losses': [],
            'G_losses': [],
            'per_epoch_ptimes': [],
            'total_ptime': [],
            'test_loss': {'mse': [], 'psnr': [], 'ssim': []}
        }

        # Get the device we're working on.
        logger.debug("cuda & device")
        logger.debug(cuda)
        logger.debug(device)

        # create sc scenarios from 0000 to 1111, yet 0000 and 1111 should not be used
        scenarios = list(map(list, itertools.product([0, 1], repeat=4)))
        scenarios.remove([0, 0, 0, 0])
        scenarios.remove([1, 1, 1, 1])

        # sort the scenarios according to decreasing difficulty. Easy scenarios last, and difficult ones first.
        scenarios.sort(key=lambda x: x.count(1))

        # crate label list
        label_list = torch.from_numpy(np.ones((BATCH_SIZE, patch[0], patch[1], patch[2]))).cuda().type(tensor) \
            if cuda else torch.from_numpy(np.ones((BATCH_SIZE, patch[0], patch[1], patch[2]))).type(tensor)

        logger.info("Start Training!")
        start_time = time.time()

        for epoch in range(START_EPOCH, RUN_EPOCHS):
            D_losses = []
            D_real_losses = []
            D_fake_losses = []
            G_train_l1_losses = []
            G_train_losses = []
            G_losses = []
            synth_losses = []

            # patient: Whole patient dictionary containing image, seg, name etc.
            # x_patient: Just the images of a single patient
            # x_r: Batch of images taken from x_patient according to the batch size specified.
            # x_z: Batch from x_r where some sequences are imputed with noise for input to G
            epoch_start_time = time.time()
            for idx_pat, patient in enumerate(n_dataloder):
                if idx_pat % 5 == 0:
                    end_num = idx_pat + 5 if idx_pat + 5 < train_patient_num else train_patient_num
                    logger.info("Epoch [%d/%d] Training patient num: %d - %d"
                                % (epoch + 1, RUN_EPOCHS, idx_pat + 1, end_num))
                # Put the whole patient in GPU to aid quicker training
                x_patient = patient['image']
                batch_indices = list(range(0, SIZE_AFTER_CROPPING[2], BATCH_SIZE))

                # this shuffles the 2D axial slice batches for efficient training
                # tag1
                random.shuffle(batch_indices)

                # create batches out of this patient
                for _num, batch_idx in enumerate(batch_indices):
                    # print("Patient #{}, Batch #{}".format(idx_pat, _num))

                    # print("\tSplicing batch from x_real")
                    x_r = x_patient[batch_idx:batch_idx + BATCH_SIZE, ...].cuda().type(tensor) if cuda \
                        else x_patient[batch_idx:batch_idx + BATCH_SIZE, ...].type(tensor)

                    rand_val = None
                    if CURRICULUM_LEARNING:
                        # Curriculum Learning:
                        # Train with easier cases in the first epochs, then start training on harder ones
                        if epoch <= 10:
                            curr_scenario_range = [11, 14]
                            rand_val = torch.randint(low=10, high=14, size=(1,))
                        if 10 < epoch <= 20:
                            curr_scenario_range = [7, 14]
                            rand_val = torch.randint(low=7, high=14, size=(1,))
                        if 20 < epoch <= 30:
                            curr_scenario_range = [3, 14]
                            rand_val = torch.randint(low=3, high=14, size=(1,))
                        if epoch > 30:
                            curr_scenario_range = [0, 14]
                            rand_val = torch.randint(low=0, high=14, size=(1,))
                    else:  # not going to take curriculum learning?
                        rand_val = torch.randint(low=0, high=14, size=(1,))

                    label_scenario = scenarios[int(rand_val.numpy()[0])]
                    # print(label_scenario)
                    logger.debug('\tTraining this batch with Scenario: {}'.format(label_scenario))

                    # create a new x_imputed and x_real with this label scenario
                    x_z = x_r.clone().cuda() if cuda else x_r.clone()
                    # x_z = x_r.clone()

                    label_list_r \
                        = torch.from_numpy(np.ones((BATCH_SIZE, patch[0], patch[1], patch[2]))).cuda().type(tensor) \
                        if cuda else torch.from_numpy(np.ones((BATCH_SIZE, patch[0], patch[1], patch[2]))).type(tensor)
                    # label_list_r.cuda()

                    impute_tensor = None
                    if IMPUTATION == 'noise':
                        impute_tensor = torch.randn((BATCH_SIZE, SPATIAL_SIZE_FOR_TRAINING[0],
                                                     SPATIAL_SIZE_FOR_TRAINING[1]), device=device)
                    elif IMPUTATION == 'average':
                        avail_indx = [i for i, x in enumerate(label_scenario) if x == 1]
                        impute_tensor = torch.mean(x_r[:, avail_indx, ...], dim=1)
                    elif IMPUTATION == 'zeros':
                        impute_tensor = torch.zeros((BATCH_SIZE, SPATIAL_SIZE_FOR_TRAINING[0],
                                                     SPATIAL_SIZE_FOR_TRAINING[1]), device=device)

                    for idx, k in enumerate(label_scenario):
                        if k == 0:
                            # print(x_z.shape, impute_tensor.shape)
                            x_z[:, idx, ...] = impute_tensor
                            # this works with both discriminator types.
                            label_list[:, idx] = 0

                        elif k == 1:
                            # this works with both discriminator types.
                            label_list[:, idx] = 1

                    # TRAIN GENERATOR G
                    # print('\tTraining Generator')
                    generator.zero_grad()
                    optimizer_G.zero_grad()

                    # print("shape of impute_tensor: ")
                    # print(impute_tensor.size())
                    # print("shape of x_z: ")
                    # print(x_z.size())
                    '''
                    show_img.show_img((x_z.cpu())[0][0])
                    show_img.show_img((x_z.cpu())[0][1])
                    show_img.show_img((x_z.cpu())[0][2])
                    show_img.show_img((x_z.cpu())[0][3])
                    '''
                    fake_x = generator(x_z)
                    # print("shape of fake_x: ")
                    # print(fake_x.shape)
                    '''
                    show_img.show_img((fake_x.cpu().detach().numpy())[0][0])
                    show_img.show_img((fake_x.cpu().detach().numpy())[0][1])
                    show_img.show_img((fake_x.cpu().detach().numpy())[0][2])
                    show_img.show_img((fake_x.cpu().detach().numpy())[0][3])
                    '''
                    # show_img.show_img((fake_x.cpu().detach().numpy())[0, :, :, :])
                    # scipy.misc.imsave('outfile.jpg', (fake_x.cpu().detach().numpy())[0][0])
                    # scipy.misc.imsave('outfile.jpg', (fake_x.cpu().detach().numpy())[0][0])
                    # scipy.misc.imsave('outfile.jpg', (fake_x.cpu().detach().numpy())[0][0])
                    # scipy.misc.imsave('outfile.jpg', (fake_x.cpu().detach().numpy())[0][0])

                    # tag1
                    if IMPLICIT_CONDITIONING:  # we're using IC
                        fake_x = impute_reals_into_fake(x_z, fake_x, label_scenario, cuda=cuda)
                        '''
                        show_img.show_img((fake_x.cpu().detach().numpy())[0][0])
                        show_img.show_img((fake_x.cpu().detach().numpy())[0][1])
                        show_img.show_img((fake_x.cpu().detach().numpy())[0][2])
                        show_img.show_img((fake_x.cpu().detach().numpy())[0][3])
    
                        show_img.show_img((x_r.cpu().detach().numpy())[0][0])
                        show_img.show_img((x_r.cpu().detach().numpy())[0][1])
                        show_img.show_img((x_r.cpu().detach().numpy())[0][2])
                        show_img.show_img((x_r.cpu().detach().numpy())[0][3])
                        '''

                    pred_fake = discriminator(fake_x, x_r)
                    '''
                    show_img.show_img((pred_fake.cpu().detach().numpy())[0][0])
                    show_img.show_img((pred_fake.cpu().detach().numpy())[0][1])
                    show_img.show_img((pred_fake.cpu().detach().numpy())[0][2])
                    show_img.show_img((pred_fake.cpu().detach().numpy())[0][3])
                    '''

                    # The discriminator should think that the pred_fake is real,
                    # so we minimize the loss between pred_fake and label_list_r, ie. make the pred_fake look real,
                    # and reducing the error that the discriminator makes when predicting it.

                    if pred_fake.size() != label_list_r.size():
                        print('Error!')
                        import sys
                        sys.exit(-1)

                    # pred_fake.cuda()
                    # pred_fake = pred_fake.to(device=device)
                    # label_list_r = label_list_r.to(device=device)

                    loss_GAN = criterion_GAN(pred_fake, label_list_r)
                    # loss_GAN.backward()

                    # pixel-wise loss
                    if IMPLICIT_CONDITIONING:
                        loss_pixel = 0
                        synth_loss = 0
                        count = 0
                        for idx_curr_label, i in enumerate(label_scenario):
                            if i == 0:
                                loss_pixel = loss_pixel + criterion_pixelwise(fake_x[:, idx_curr_label, ...],
                                                                              x_r[:, idx_curr_label, ...])

                                synth_loss = synth_loss + mse_fake_vs_real(fake_x[:, idx_curr_label, ...],
                                                                           x_r[:, idx_curr_label, ...])
                                count = count + 1

                        loss_pixel = loss_pixel / count
                        synth_loss = loss_pixel / count
                    else:  # no IC, calculate loss for all output w.r.t all GT.
                        loss_pixel = criterion_pixelwise(fake_x, x_r)

                        synth_loss = mse_fake_vs_real(fake_x, x_r)

                    # LAMBDA = 0.9 - variable that sets the relative importance to loss_GAN and loss_pixel
                    G_train_total_loss = (1 - LAMBDA) * loss_GAN + LAMBDA * loss_pixel

                    # loss_GAN.backward()
                    # loss_pixel.backward()

                    G_train_total_loss.backward()
                    optimizer_G.step()

                    # save the losses
                    # print(loss_pixel.item())
                    # print(loss_pixel.shape)
                    G_train_l1_losses.append(loss_pixel.item())
                    G_train_losses.append(loss_GAN.item())
                    G_losses.append(G_train_total_loss.item())
                    synth_losses.append(synth_loss.item())

                    # TRAIN DISCRIMINATOR D
                    # this takes in the real x as X-INPUT and real x as Y-INPUT
                    # print('\tTraining Discriminator')
                    discriminator.zero_grad()
                    optimizer_D.zero_grad()

                    # real loss
                    # EDIT: We removed noise addition
                    # We can add noise to the inputs of the discriminator
                    pred_real = discriminator(x_r, x_r)

                    loss_real = criterion_GAN(pred_real, label_list_r)

                    # fake loss
                    # fake_x = generator(x_z, label_map)
                    fake_x = generator(x_z)

                    # tag1
                    if IMPLICIT_CONDITIONING:
                        fake_x = impute_reals_into_fake(x_z, fake_x, label_scenario)

                    # we add noise to the inputs of the discriminator here as well
                    pred_fake = discriminator(fake_x.detach(), x_r)
                    # pred_fake = discriminator(fake_x, x_r)

                    loss_fake = criterion_GAN(pred_fake, label_list)

                    D_train_loss = 0.5 * (loss_real + loss_fake)

                    # for printing purposes
                    D_real_losses.append(loss_real.item())
                    D_fake_losses.append(loss_fake.item())
                    D_losses.append(D_train_loss.item())

                    D_train_loss.backward()
                    optimizer_D.step()

                    logger.debug(" E [{}/{}] P #{} ".format(epoch + 1, RUN_EPOCHS, idx_pat)
                          + 'B [%d/%d] - loss_d: [real: %.5f, fake: %.5f, comb: %.5f], loss_g: '
                            '[gan: %.5f, l1: %.5f, comb: %.5f], synth_loss_mse(ut): %.5f'
                          % ((_num + 1), SIZE_AFTER_CROPPING[2] // BATCH_SIZE,
                             torch.mean(torch.FloatTensor(D_real_losses)),
                             torch.mean(torch.FloatTensor(D_fake_losses)),
                             torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_train_losses)),
                             torch.mean(torch.FloatTensor(G_train_l1_losses)),
                             torch.mean(torch.FloatTensor(G_losses)),
                             torch.mean(torch.FloatTensor(synth_losses))))

                # Check if we have trained with exactly opt.train_patient_idx patients
                # (if opt.train_patient_idx is 10, then idx_pat will be 9, so this condition will evaluate to true
                if idx_pat + 1 == train_patient_num:
                    logger.info('Testing on test set for this fold')
                    # main_path = os.path.join(TOP_LEVEL_PATH, data_type, "{}".format("BRAST2018"), 'scenario_results')

                    logger.debug("Saving results at {}".format(results_path))

                    generator.eval()

                    logger.info("Calculating metric on test set")
                    result_dict_test, _running_mse, _running_psnr, _running_ssim = calculate_metrics(
                        generator, test_patient, save_path=results_path,
                        all_scenarios=copy.deepcopy(scenarios), epoch=epoch, save_stats=True,
                        curr_scenario_range=None, batch_size_to_test=TEST_BATCH_SIZE,
                        impute_type=IMPUTATION, cuda=cuda)

                    logger.info("\tTesting Performance Numbers")
                    printTable(result_dict_test)
                    gc.collect()

                    generator.train()
                    gc.collect()
                    break

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            logger.info('[%d/%d] - ptime: %.2f, loss_d: [real: %.5f, fake: %.5f, comb: %.5f], loss_g: '
                  '[gan: %.5f, l1: %.5f, comb: %.5f], synth_loss_mse(ut): %.5f' % (
                      (epoch + 1), RUN_EPOCHS, per_epoch_ptime, torch.mean(torch.FloatTensor(D_real_losses)),
                      torch.mean(torch.FloatTensor(D_fake_losses)),
                      torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_train_losses)),
                      torch.mean(torch.FloatTensor(G_train_l1_losses)), torch.mean(torch.FloatTensor(G_losses)),
                      torch.mean(torch.FloatTensor(synth_losses))))

            # Checkpoint the models

            gen_state_checkpoint = {
                'epoch': epoch + 1,
                'arch': ARCH,
                'state_dict': generator.state_dict(),
                'optimizer': optimizer_G.state_dict(),
            }

            des_state_checkpoint = {
                'epoch': epoch + 1,
                'arch': ARCH,
                'state_dict': discriminator.state_dict(),
                'optimizer': optimizer_D.state_dict(),
            }

            if not os.path.isdir(os.path.join(TOP_LEVEL_PATH, 'out', dataset_name)):
                os.mkdir(os.path.join(TOP_LEVEL_PATH, 'out', dataset_name))

            if (epoch + 1) % 10 in SAVE_PARAMETERS_EPOCH or epoch == RUN_EPOCHS - 1:  # the parameters are really large!
                save_checkpoint(gen_state_checkpoint,
                                os.path.join(TOP_LEVEL_PATH, 'out', dataset_name,
                                             'generator_param_{}_{}.pkl'.format(dataset_name, epoch + 1)),
                                pickle_module=pickle)

                save_checkpoint(des_state_checkpoint,
                                os.path.join(TOP_LEVEL_PATH, 'out', dataset_name,
                                             'discriminator_param_{}_{}.pkl'.format(dataset_name, epoch + 1)),
                                pickle_module=pickle)

            if result_dict_test is not None:
                with open(os.path.join(TOP_LEVEL_PATH, 'out', dataset_name,
                                       'result_dict_test_epoch_{}.pkl'.format(epoch)), 'wb') as f:
                    pickle.dump(result_dict_test, f)

                logger.info('[Testing] num_pats: {}, mse: {:.5f}, psnr: {:.5f}, ssim: {:.5f}'.format(
                    test_patient_num,
                    result_dict_test['mean']['mse'],
                    result_dict_test['mean']['psnr'],
                    result_dict_test['mean']['ssim']
                ))

                train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
                train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

                train_hist['test_loss']['mse'].append(result_dict_test['mean']['mse'])
                train_hist['test_loss']['psnr'].append(result_dict_test['mean']['psnr'])
                train_hist['test_loss']['ssim'].append(result_dict_test['mean']['ssim'])

                train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

            end_time = time.time()
            total_ptime = end_time - start_time
            train_hist['total_ptime'].append(total_ptime)

            # logger.info("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
            #     torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), RUN_EPOCHS, total_ptime))

            logger.info("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
                total_ptime / (epoch + 1 - START_EPOCH), epoch + 1 - START_EPOCH, total_ptime))

        with open(os.path.join(TOP_LEVEL_PATH, 'out', dataset_name, 'train_hist.pkl'), 'wb') as f:
            pickle.dump(train_hist, f)


if __name__ == '__main__':
    train(DATASET_PATH, 'HGG', TRAIN_PATIENTS_HGG, TEST_PATIENTS_HGG)
    train(DATASET_PATH, 'LGG', TRAIN_PATIENTS_LGG, TEST_PATIENTS_LGG)
