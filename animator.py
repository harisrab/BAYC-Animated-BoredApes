# Imports
import glob
import os
from src.approaches.train_audio2landmark import Audio2landmark_model
import shutil
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
import pickle
import argparse
import numpy as np
import sys
sys.path.append('thirdparty/AdaptiveWingLoss')


# Global Flags
ADD_NAIVE_EYE = False
GEN_AUDIO = True
GEN_FLS = True

DEMO_CH = 'ape2.jpg'


class Animator():
    def __init__(self, inputImage, inputAudio, outputFN="out.mp4", outputFolder="ape_src", audio_dir="audio"):
        self.args = {
            'jpg': f'{inputImage}.jpg',
            'jpg_bg': f'{inputImage}_bg.jpg',
            'inner_lip': False,
            'out': outputFN,
            'load_AUTOVC_name': 'examples/ckpt/ckpt_autovc.pth',
            'load_a2l_G_name': 'examples/ckpt/ckpt_speaker_branch.pth',
            'load_a2l_C_name': 'examples/ckpt/ckpt_content_branch.pth',
            'load_G_name': 'examples/ckpt/ckpt_116_i2i_comb.pth',
            'amp_lip_x': 2.0,
            'amp_lip_y': 2.0,
            'amp_pos': 0.5,
            'reuse_train_emb_list': [],
            'add_audio_in': False,
            'comb_fan_awing': False,
            'output_folder': outputFolder,
            'img_input_dir': outputFolder,
            'test_end2end': True,
            'dump_dir': '',
            'pos_dim': 7,
            'use_prior_net': True,
            'transformer_d_model': 32,
            'transformer_N': 2,
            'transformer_heads': 2,
            'spk_emb_enc_size': 16,
            'init_content_encoder': '',
            'lr': 0.001,
            'reg_lr': 1e-06,
            'write': False,
            'segment_batch_size': 512,
            'emb_coef': 3.0,
            'lambda_laplacian_smooth_loss': 1.0,
            'use_11spk_only': False,
            'audio_input_directory': audio_dir,
        }

        # Input Image
        self.DEMO_CHARACTER = self.args.jpg.split('.')[0]

        # Load Closed Mouth Facial Landmarks
        self.face_shape = np.loadtxt(
            f'ape_src/{self.DEMO_CHARACTER}_face_close_mouth.txt')

    def GenerateAudioInput(self):
        """ This function returns all input audio file names, corresponding embeddings, and embeddings for samples"""

        au_data = []
        au_emb = []

        # List of input audio files
        ains = glob.glob1(self.args.audio_input_directory, '*.wav')

        # Remove temporary tmp.wav file
        ains = [item for item in ains if item is not 'tmp.wav']

        # Sort audio input filenames
        ains.sort()

        for eachAudio in ains:
            # For each audio input change the sample frequency to 16000 Hz and save in audio_dir/tmp.wav
            os.system(
                f'ffmpeg -y -loglevel error -i {self.args.audio_input_directory}/{eachAudio} -ar 16000 {self.args.audio_input_directory}/tmp.wav')

            # Replace old audio with new one of new sampling frequency
            shutil.copyfile(f'{self.args.audio_input_directory}/tmp.wav',
                            f'{self.args.audio_input_directory}/{eachAudio}')

            # Generate embedding for eachAudio
            mean_emb, all_emb = get_spk_emb(
                f'{self.args.audio_input_directory}/{eachAudio}')

            # Append to list of audio embeddings
            au_emb.append(mean_emb.reshape(-1))

            print("---------------------------------")
            print('Processing audio file', eachAudio)
            print("---------------------------------")

            # Generate embeddings for each sample for each eachAudio
            c = AutoVC_mel_Convertor(self.args.audio_input_directory)
            au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=os.path.join(self.args.audio_input_directory, eachAudio),
                                                             autovc_model_path=self.args.load_AUTOVC_name)
            au_data += au_data_i

        # If tmp.wav still exists, remove it
        if(os.path.isfile(f'{self.args.audio_input_directory}/tmp.wav')):
            os.remove(f'{self.args.audio_input_directory}/tmp.wav')

        return {"audio_inputs": ains, "audio_input_emb": au_emb, "audio_sample_emb": au_data}

    def GetFacialLandmarkData(self, au_data):
        fl_data = []
        audio_dir = self.args.audio_input_directory

        rot_tran, rot_quat, anchor_t_shape = [], [], []

        for au, info in au_data:
            au_length = au.shape[0]
            fl = np.zeros(shape=(au_length, 68 * 3))
            fl_data.append((fl, info))
            rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
            rot_quat.append(np.zeros(shape=(au_length, 4)))
            anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

        # Create data dumps if they don't exist
        if(os.path.exists(os.path.join(audio_dir, 'dump', 'random_val_fl.pickle'))):
            os.remove(os.path.join(audio_dir, 'dump', 'random_val_fl.pickle'))

        if(os.path.exists(os.path.join(audio_dir, 'dump', 'random_val_fl_interp.pickle'))):
            os.remove(os.path.join(audio_dir, 'dump',
                      'random_val_fl_interp.pickle'))

        if(os.path.exists(os.path.join(audio_dir, 'dump', 'random_val_au.pickle'))):
            os.remove(os.path.join(audio_dir, 'dump', 'random_val_au.pickle'))

        if (os.path.exists(os.path.join(audio_dir, 'dump', 'random_val_gaze.pickle'))):
            os.remove(os.path.join(audio_dir, 'dump', 'random_val_gaze.pickle'))

        # Pack the landmark data, audio_embedding data, and gaze data
        with open(os.path.join(audio_dir, 'dump', 'random_val_fl.pickle'), 'wb') as fp:
            pickle.dump(fl_data, fp)

        with open(os.path.join(audio_dir, 'dump', 'random_val_au.pickle'), 'wb') as fp:
            pickle.dump(au_data, fp)

        with open(os.path.join(audio_dir, 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
            gaze = {'rot_trans': rot_tran, 'rot_quat': rot_quat,
                    'anchor_t_shape': anchor_t_shape}

            pickle.dump(gaze, fp)

        return fl_data

    def AudioToLandmark(self, au_emb):
        ''' STEP 4: RUN audio -> landmark network'''

        model = Audio2landmark_model(self.args, jpg_shape=self.face_shape)
        if(len(self.args.reuse_train_emb_list) == 0):
            model.test(au_emb=au_emb)
        else:
            model.test(au_emb=None)
        print('Finished generating landmarks for each audio sample')

    def DenormalizeOutputToOriginalImage(self):
        fls_names = glob.glob1(self.args.img_input_dir, 'pred_fls_*.txt')
        fls_names.sort()
        print('fls_names' + fls_names)
        print("Facial Landmarks Names ====> ", fls_names)

        for i in range(0, len(fls_names)):
            # Pick all names of input audios
            ains = glob.glob1(self.args.audio_input_directory, '*.wav')
            ains.sort()

            # Take (1) input audio and corresponding (2) facial landmarks
            ain = ains[i]
            fl = np.loadtxt(os.path.join(
                self.args.img_input_dir, fls_names[i])).reshape((-1, 68, 3))

            output_dir = os.path.join('ape_src', fls_names[i][:-4])

            print("Named output directory: ", output_dir)

            try:
                os.makedirs(output_dir)

            except:
                pass

            from util.utils import get_puppet_info

            bound, scale, shift = get_puppet_info(
                self.DEMO_CHARACTER, ROOT_DIR='ape_src')

            fls = fl.reshape((-1, 68, 3))

            fls[:, :, 0:2] = -fls[:, :, 0:2]
            fls[:, :, 0:2] = (fls[:, :, 0:2] / scale)
            fls[:, :, 0:2] -= shift.reshape(1, 2)

            fls = fls.reshape(-1, 204)

            # additional smooth
            from scipy.signal import savgol_filter
            fls[:, 0:48*3] = savgol_filter(fls[:, 0:48*3], 17, 3, axis=0)
            fls[:, 48*3:] = savgol_filter(fls[:, 48*3:], 11, 3, axis=0)
            fls = fls.reshape((-1, 68, 3))

            # if (DEMO_CH in ['paint', 'mulaney', 'cartoonM', 'beer', 'color', 'JohnMulaney', 'vangogh', 'jm', 'roy', 'lineface']):
            if(not opt_parser.inner_lip):
                r = list(range(0, 68))
                fls = fls[:, r, :]
                fls = fls[:, :, 0:2].reshape(-1, 68 * 2)
                fls = np.concatenate(
                    (fls, np.tile(bound, (fls.shape[0], 1))), axis=1)
                fls = fls.reshape(-1, 160)

            else:
                r = list(range(0, 48)) + list(range(60, 68))
                fls = fls[:, r, :]
                fls = fls[:, :, 0:2].reshape(-1, 56 * 2)
                fls = np.concatenate(
                    (fls, np.tile(bound, (fls.shape[0], 1))), axis=1)
                fls = fls.reshape(-1, 112 + bound.shape[1])

            np.savetxt(os.path.join(
                output_dir, 'warped_points.txt'), fls, fmt='%.2f')

            # static_points.txt
            static_frame = np.loadtxt(os.path.join(
                'ape_src', '{}_face_open_mouth.txt'.format(DEMO_CH)))
            static_frame = static_frame[r, 0:2]
            static_frame = np.concatenate(
                (static_frame, bound.reshape(-1, 2)), axis=0)
            np.savetxt(os.path.join(output_dir, 'reference_points.txt'),
                       static_frame, fmt='%.2f')

            # triangle_vtx_index.txt
            shutil.copy(os.path.join('ape_src', DEMO_CH + '_delauney_tri.txt'),
                        os.path.join(output_dir, 'triangulation.txt'))

            os.remove(os.path.join('ape_src', fls_names[i]))

            # ==============================================
            # Step 4 : Vector art morphing
            # ==============================================
            warp_exe = os.path.join(os.getcwd(), 'facewarp', 'facewarp.exe')
            import os

            if (os.path.exists(os.path.join(output_dir, 'output'))):
                shutil.rmtree(os.path.join(output_dir, 'output'))
            os.mkdir(os.path.join(output_dir, 'output'))
            os.chdir('{}'.format(os.path.join(output_dir, 'output')))
            cur_dir = os.getcwd()
            print(cur_dir)

            if(os.name == 'nt'):
                ''' windows '''
                os.system('{} {} {} {} {} {}'.format(
                    warp_exe,
                    os.path.join(cur_dir, '..', '..', opt_parser.jpg),
                    os.path.join(cur_dir, '..', 'triangulation.txt'),
                    os.path.join(cur_dir, '..', 'reference_points.txt'),
                    os.path.join(cur_dir, '..', 'warped_points.txt'),
                    os.path.join(cur_dir, '..', '..', opt_parser.jpg_bg),
                    '-novsync -dump'))
            else:
                ''' linux '''
                os.system('wine {} {} {} {} {} {}'.format(
                    warp_exe,
                    os.path.join(cur_dir, '..', '..', opt_parser.jpg),
                    os.path.join(cur_dir, '..', 'triangulation.txt'),
                    os.path.join(cur_dir, '..', 'reference_points.txt'),
                    os.path.join(cur_dir, '..', 'warped_points.txt'),
                    os.path.join(cur_dir, '..', '..', opt_parser.jpg_bg),
                    '-novsync -dump'))

            os.system('ffmpeg -y -r 62.5 -f image2 -i "%06d.tga" -i {} -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -shortest -strict -2 {}'.format(
                os.path.join(cur_dir, '..', '..', '..', 'examples', ain),
                os.path.join(cur_dir, '..', 'out.mp4')
            ))
