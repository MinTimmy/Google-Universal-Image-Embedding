import architectures.resnet50
import architectures.bninception
import architectures.resnet50_for_submission

def select(arch, opt):
    print('arch', arch)
    if 'resnet50_for_submission' in arch:
        return resnet50_for_submission.Network(opt)
    if 'resnet50' in arch:
        return resnet50.Network(opt)
    if 'bninception' in arch:
        return bninception.Network(opt)
