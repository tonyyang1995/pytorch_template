import argparse
import os

import torch

class BaseOptions():
    # this class defines options used during both training and testing
    def __init__(self):
        self.initialized = False

    def initialized(self, parser):
        # parser.add_argument('--dataroot', default="")

        
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name + suffix')
        self.initialized = True
        return parser
    
    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultHelpFormatter)
            parser = self.initialized(parser)
        
        opt, _ = parser.parse_known_args()

        self.parser = parser
        return parser.parse_args()
        
    def print_options(self, opt):
        message = ''
        message += '------------------- Options -------------------\n'
        for k,v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '------------------- End -----------------------\n'
        print(message)
    
        exper_dir = os.path.join(opt.checkpoints_dir, opt.name)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(exper_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    
    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix 
        
        self.print_options(opt)

        # set up gpu
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        
        self.opt = opt
        return self.opt
            