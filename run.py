import argparse
import configparser
import pathlib
from distutils.util import strtobool


def set_arg_parser():
    description = "This is MM-GAN"
    parser = argparse.ArgumentParser(prog='MM-GAN',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description=description)
    parser.add_argument('--tftrain', dest='tf_train_cfg', type=str,
                        help="run training session with tensorflow2, please provide a config file, note that the saved "
                             "models for tensorflow2 is different from pytorch version")
    parser.add_argument('--torchtrain', dest='torch_train_cfg', type=str,
                        help="run training session with pytorch, please provide a config file, note that the saved "
                             "models for pytorch is different from tensorflow2 version")
    parser.add_argument('--tftest', dest='tf_test_cfg', type=str,
                        help="If the model is trained with tensorflow2, use this for testing")
    parser.add_argument('--torchtest', dest='torch_test_cfg', type=str,
                        help="If the model is trained with pytorch, use this for testing")
    parser.add_argument('--preprocess', dest='preprocess_cfg', type=str,
                        help="Use this to separate the dataset into five folds, and normalize them")
    return parser


def read_train_cfg(cfg_path: str):
    config = configparser.ConfigParser()
    config.read(cfg_path)
    cfg_dict = {}

    # read paths and names
    # get list of paths for training and validation
    cfg_dict['tfrecords_train'] = config.get("PATHS_AND_NAMES", "tfrecords_train").split()
    cfg_dict['tfrecords_val'] = config.get("PATHS_AND_NAMES", "tfrecords_val").split()
    cfg_dict['session_name'] = config.get("PATHS_AND_NAMES", "session_name")
    cfg_dict['output_dir'] = config.get("PATHS_AND_NAMES", "output_dir")

    # read training strategy
    cfg_dict['epochs'] = int(config.get("STRATEGY", "epochs"))
    cfg_dict['epochs_per_dom_epoch'] = int(config.get("STRATEGY", "epochs_per_dom_epoch"))
    cfg_dict['sub_epochs'] = int(config.get("STRATEGY", "sub_epochs"))
    cfg_dict['batch_size_train'] = int(config.get("STRATEGY", "batch_size_train"))
    cfg_dict['full_random'] = bool(strtobool(config.get("STRATEGY", "full_random")))
    cfg_dict['implicit_conditioning'] = bool(strtobool(config.get("STRATEGY", "implicit_conditioning")))
    cfg_dict['curriculum_learning'] = bool(strtobool(config.get("STRATEGY", "curriculum_learning")))
    cfg_dict['focus_on_worst'] = bool(strtobool(config.get("STRATEGY", "focus_on_worst")))
    cfg_dict['debug'] = bool(strtobool(config.get("STRATEGY", "debug")))

    img_shape = config.get("HIPPER_PARAMS", "img_shape").split()
    cfg_dict['img_shape'] = (int(img_shape[0]), int(img_shape[1]))
    cfg_dict['learning_rate'] = float(config.get("HIPPER_PARAMS", "learning_rate"))
    cfg_dict['beta_1'] = float(config.get("HIPPER_PARAMS", "beta_1"))
    cfg_dict['beta_2'] = float(config.get("HIPPER_PARAMS", "beta_2"))
    cfg_dict['lambda_param'] = float(config.get("HIPPER_PARAMS", "lambda_param"))

    return cfg_dict


def read_test_cfg(cfg_path: str):
    config = configparser.ConfigParser()
    config.read(cfg_path)
    cfg_dict = {}

    # read paths and names
    # get list of paths for training and validation
    cfg_dict['tfrecords_test'] = config.get("PATHS_AND_NAMES", "tfrecords_test").split()
    cfg_dict['channels'] = config.get("PATHS_AND_NAMES", "channels").split()
    cfg_dict['session_name'] = config.get("PATHS_AND_NAMES", "session_name")
    cfg_dict['output_dir'] = config.get("PATHS_AND_NAMES", "output_dir")
    cfg_dict['parameter_path'] = config.get("PATHS_AND_NAMES", "parameter_path")

    img_shape = config.get("HIPPER_PARAMS", "img_shape").split()
    cfg_dict['img_shape'] = (int(img_shape[0]), int(img_shape[1]))

    return cfg_dict


def read_preprocess_cfg(cfg_path: str):
    config = configparser.ConfigParser()
    config.read(cfg_path)
    cfg_dict = {}

    # read paths and names
    # get list of paths for training and validation
    cfg_dict['channels'] = config.get("PATHS_AND_NAMES", "channels").split()
    cfg_dict['group_txt_names'] = config.get("PATHS_AND_NAMES", "group_txt_names").split()
    cfg_dict['group_txt_out_names'] = config.get("PATHS_AND_NAMES", "group_txt_out_names").split()
    cfg_dict['raw_data_dir'] = config.get("PATHS_AND_NAMES", "raw_data_dir")
    cfg_dict['output_dir'] = config.get("PATHS_AND_NAMES", "output_dir")

    cfg_dict['new_five_folds'] = bool(strtobool(config.get("STRATEGY", "new_five_folds")))
    slice_range = config.get("STRATEGY", "slice_range").split()
    cfg_dict['slice_range'] = (int(slice_range[0]), int(slice_range[1]))
    cfg_dict['shuffle'] = bool(strtobool(config.get("STRATEGY", "shuffle")))
    cfg_dict['operation'] = config.get("STRATEGY", "operation")
    crop_cord = config.get("STRATEGY", "crop_cord").split()
    cfg_dict['crop_cord'] = (int(crop_cord[0]), int(crop_cord[1]), int(crop_cord[2]), int(crop_cord[3]))

    return cfg_dict


def main():
    parser = set_arg_parser()
    args = parser.parse_args()
    print(args)

    # todo: remember to check inputs

    cfg_path, test_cfg_path = None, None

    if args.tf_train_cfg:
        cfg_path = args.tf_train_cfg
        path = pathlib.Path(cfg_path)
        if not path.is_file():
            print("Config file: %s dose not exist!" % cfg_path)
        else:
            cfg_dict = read_train_cfg(cfg_path)
            print(cfg_dict)
            import mm_gan.tf_train as tf_train
            tf_train.train(session_name=cfg_dict['session_name'], output_dir=cfg_dict['output_dir'],
                           tfrecords_train=cfg_dict['tfrecords_train'], tfrecords_val=cfg_dict['tfrecords_val'],
                           batch_size_train=cfg_dict['batch_size_train'], full_random=cfg_dict['full_random'],
                           img_shape=cfg_dict['img_shape'], learning_rate=cfg_dict['learning_rate'],
                           beta_1=cfg_dict['beta_1'], beta_2=cfg_dict['beta_2'],
                           lambda_param=cfg_dict['lambda_param'], epochs=cfg_dict['epochs'],
                           epochs_per_dom_epoch=cfg_dict['epochs_per_dom_epoch'],
                           sub_epochs=cfg_dict['sub_epochs'], implicit_conditioning=cfg_dict['implicit_conditioning'],
                           curriculum_learning=cfg_dict['curriculum_learning'],
                           focus_on_worst=cfg_dict['focus_on_worst'],
                           debug=cfg_dict['debug'])
    elif args.torch_train_cfg:
        cfg_path = args.torch_train_cfg
        path = pathlib.Path(cfg_path)
        if not path.is_file():
            print("Config file: %s dose not exist!" % cfg_path)
        else:
            cfg_dict = read_train_cfg(cfg_path)
            print(cfg_dict)
            import mm_gan.torch_train as torch_train
            torch_train.train(session_name=cfg_dict['session_name'], output_dir=cfg_dict['output_dir'],
                              tfrecords_train=cfg_dict['tfrecords_train'], tfrecords_val=cfg_dict['tfrecords_val'],
                              batch_size_train=cfg_dict['batch_size_train'], full_random=cfg_dict['full_random'],
                              img_shape=cfg_dict['img_shape'], learning_rate=cfg_dict['learning_rate'],
                              beta_1=cfg_dict['beta_1'], beta_2=cfg_dict['beta_2'],
                              lambda_param=cfg_dict['lambda_param'], epochs=cfg_dict['epochs'],
                              epochs_per_dom_epoch=cfg_dict['epochs_per_dom_epoch'],
                              sub_epochs=cfg_dict['sub_epochs'],
                              implicit_conditioning=cfg_dict['implicit_conditioning'],
                              curriculum_learning=cfg_dict['curriculum_learning'],
                              focus_on_worst=cfg_dict['focus_on_worst'],
                              debug=cfg_dict['debug'])
    elif args.tf_test_cfg:
        cfg_path = args.tf_test_cfg
        path = pathlib.Path(cfg_path)
        if not path.is_file():
            print("Config file: %s dose not exist!" % cfg_path)
        else:
            cfg_dict = read_test_cfg(cfg_path)
            print(cfg_dict)
            import mm_gan.test as test
            test.tf_test(parameter_path=cfg_dict['parameter_path'], session_name=cfg_dict['session_name'],
                         output_dir=cfg_dict['output_dir'], tfrecords=cfg_dict['tfrecords_test'],
                         channels=cfg_dict['channels'], img_shape=cfg_dict['img_shape'])
    elif args.torch_test_cfg:
        cfg_path = args.torch_test_cfg
        path = pathlib.Path(cfg_path)
        if not path.is_file():
            print("Config file: %s dose not exist!" % cfg_path)
        else:
            cfg_dict = read_test_cfg(cfg_path)
            print(cfg_dict)
            import mm_gan.test as test
            test.torch_test(parameter_path=cfg_dict['parameter_path'], session_name=cfg_dict['session_name'],
                            output_dir=cfg_dict['output_dir'], tfrecords=cfg_dict['tfrecords_test'],
                            channels=cfg_dict['channels'], img_shape=cfg_dict['img_shape'])
    elif args.preprocess_cfg:
        cfg_path = args.preprocess_cfg
        path = pathlib.Path(cfg_path)
        if not path.is_file():
            print("Config file: %s dose not exist!" % cfg_path)
        else:
            cfg_dict = read_preprocess_cfg(cfg_path)
            print(cfg_dict)
            import mm_gan.preprocess as preprocess
            if cfg_dict['new_five_folds']:
                preprocess.arrange_data_into_5_folds(
                    parent_path=cfg_dict['raw_data_dir'],
                    output_dir=cfg_dict['output_dir'],
                    shuffle=cfg_dict['shuffle']
                )
            preprocess.preprocess(group_text_dir=cfg_dict['output_dir'], group_text_names=cfg_dict['group_txt_names'],
                                  output_root_dir=cfg_dict['output_dir'], channels=cfg_dict['channels'])
            preprocess.nii_to_tfrecord(txt_path=cfg_dict['output_dir'], txt_names=cfg_dict['group_txt_out_names'],
                                       output_dir=cfg_dict['output_dir'], channels=cfg_dict['channels'],
                                       operation=cfg_dict['operation'], slice_range=cfg_dict['slice_range'],
                                       crop_cord=cfg_dict['crop_cord'],)
            preprocess.move_segmasks_to_folds(raw_group_txt_dir=cfg_dict['output_dir'],
                                              raw_group_txt_names=cfg_dict['group_txt_names'],
                                              std_group_txt_dir=cfg_dict['output_dir'],
                                              std_group_txt_names=cfg_dict['group_txt_out_names'])
            preprocess.move_segmasks_to_folds_and_remove_some_slices(
                raw_group_txt_dir=cfg_dict['output_dir'], raw_group_txt_names=cfg_dict['group_txt_names'],
                std_group_txt_dir=cfg_dict['output_dir'], std_group_txt_names=cfg_dict['group_txt_out_names'],
                slices_range=cfg_dict['slice_range'],
            )


if __name__ == '__main__':
    main()
