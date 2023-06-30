import tensorflow as tf
import numpy as np
import os 
import subprocess
import shutil
import hashlib
import argparse
import sys
import glob


MOBILNETV3_MINI_TFLITE_FILENAME         = 'v3-large-minimalistic_224_1.0_float.tflite'
MOBILNETV3_MINI_LBL_FILENAME            = 'imagenet_1000_labels.txt'
MOBILNETV3_MINI_DLC_FILENAME            = 'v3-large-minimalistic_224_1.0_float.dlc'
MOBILNETV3_MINI_DLC_QUANTIZED_FILENAME  = 'v3-large-minimalistic_224_1.0_quantized.dlc'
RAW_LIST_FILE                           = 'raw_list.txt'
TARGET_RAW_LIST_FILE                    = 'target_raw_list.txt'

def prepare_data_images(snpe_root, model_dir, tensorflowlite_dir):
    # make a copy of the image files from the alexnet model data dir
    src_img_files = os.path.join(snpe_root, 'models', 'alexnet', 'data', '*.jpg')
    data_dir = os.path.join(model_dir, 'data')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        os.makedirs(data_dir + '/cropped')
    for file in glob.glob(src_img_files):
        shutil.copy(file, data_dir)

    # copy the labels file to the data directory
    src_label_file = os.path.join(tensorflowlite_dir, MOBILNETV3_MINI_LBL_FILENAME)
    shutil.copy(src_label_file, data_dir)

    print('INFO: Creating SNPE mobilnetv3_mini raw data')
    scripts_dir = os.path.join(model_dir, 'scripts')
    create_raws_script = os.path.join(scripts_dir, 'create_mobilenetv3_mini_raws.py')
    data_cropped_dir = os.path.join(data_dir, 'cropped')
    print(create_raws_script)
    print(data_cropped_dir)
    cmd = ['python', create_raws_script,
           '-i', data_dir,
           '-d',data_cropped_dir]
    subprocess.call(cmd)
    print(create_raws_script)
    print(data_cropped_dir)

    print('INFO: Creating image list data files')
    create_file_list_script = os.path.join(scripts_dir, 'create_file_list.py')
    cmd = ['python', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_cropped_dir, RAW_LIST_FILE),
           '-e', '*.raw']
    subprocess.call(cmd)
    cmd = ['python', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_dir, TARGET_RAW_LIST_FILE),
           '-e', '*.raw',
           '-r']
    subprocess.call(cmd)


def convert_to_dlc(tflite_filename, model_dir, tensorflowlite_dir, runtime, htp_soc):
    print('INFO: Converting ' + tflite_filename +' to SNPE DLC format')
    dlc_dir = os.path.join(model_dir, 'dlc')
    if not os.path.isdir(dlc_dir):
        os.makedirs(dlc_dir)
    dlc_name = MOBILNETV3_MINI_DLC_FILENAME
    cmd = ['snpe-tflite-to-dlc',
           '--input_network', os.path.join(tensorflowlite_dir, tflite_filename),
           '--input_dim', 'input', '1,224,224,3',
           '--out_node', 'MobilenetV3/Predictions/Softmax',
           '--output_path', os.path.join(dlc_dir, dlc_name)]
    subprocess.call(cmd)

    # Further optimize the model with quantization for fixed-point runtimes if required.
    if ('dsp' == runtime or 'aip' == runtime or 'all' == runtime):
        quant_dlc_name = MOBILNETV3_MINI_DLC_QUANTIZED_FILENAME
        print('INFO: Creating ' + quant_dlc_name + ' quantized model')
        data_cropped_dir = os.path.join(os.path.join(model_dir, 'data'), 'cropped')
        cmd = ['snpe-dlc-quantize',
               '--input_dlc', os.path.join(dlc_dir, dlc_name),
               '--input_list', os.path.join(data_cropped_dir, RAW_LIST_FILE),
               '--output_dlc', os.path.join(dlc_dir, quant_dlc_name)]
        if (htp_soc and ('dsp' == runtime or "all" == runtime)):
            print ('INFO: Compiling HTP metadata for DSP runtime.')
            cmd.append('--enable_htp')
            cmd.append('--htp_socs')
            cmd.append(htp_soc)
        elif ('aip' == runtime or "all" == runtime):
            print ('INFO: Compiling HTA metadata for AIP runtime.')
            # Enable compilation on the HTA after quantization
            cmd.append ('--enable_hta')
        subprocess.call(cmd , env=os.environ)

def setup_assets(runtime, htp_soc):

    if 'SNPE_ROOT' not in os.environ:
        raise RuntimeError('SNPE_ROOT not setup.  Please run the SDK env setup script.')

    snpe_root = os.path.abspath(os.environ['SNPE_ROOT'])
    if not os.path.isdir(snpe_root):
        raise RuntimeError('SNPE_ROOT (%s) is not a dir' % snpe_root)

    if None == runtime:
        # No runtime specified. Use cpu as default runtime
        runtime = 'cpu'
    runtimes_list = ['cpu', 'gpu', 'dsp', 'aip', 'all']
    if runtime not in runtimes_list:
        raise RuntimeError('%s not a valid runtime. See help.' % runtime)
    
    model_dir = os.path.join(snpe_root, 'models', 'mobilenet_v3_minimalistic')
    if not os.path.isdir(model_dir):
        raise RuntimeError('%s does not exist.  Your SDK may be faulty.' % model_dir)

    tensorflowlite_dir = os.path.join(model_dir, 'tensorflowlite')
    if not os.path.isdir(tensorflowlite_dir):
        os.makedirs(tensorflowlite_dir)

    tflite_filename = MOBILNETV3_MINI_TFLITE_FILENAME

    prepare_data_images(snpe_root, model_dir, tensorflowlite_dir)

    convert_to_dlc(tflite_filename, model_dir, tensorflowlite_dir, runtime, htp_soc)
    print('INFO: Setup mobilenet v3 mini completed.')

def getArgs():

    parser = argparse.ArgumentParser(
        prog=__file__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
        '''Prepares the mobilenetv3 assets for tutorial examples.''')

    parser._action_groups.pop()
    optional = parser.add_argument_group('optional arguments')

    optional.add_argument('-r', '--runtime', type=str, required=False,
                        help='Choose a runtime to set up tutorial for. Choices: cpu, gpu, dsp, aip, all. \'all\' option is only supported with --udo flag')
    optional.add_argument('-l', '--htp_soc', type=str, nargs='?', const='sm8550', required=False,
                        help='Specify SOC target for generating HTP Offline Cache. For example: "--htp_soc sm8450" for waipio, default value is sm8550')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = getArgs()

    try:
        setup_assets(args.runtime, args.htp_soc)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))