import abc
import torch
from torch.utils.data import DataLoader
from lib.dataset import SingleFaceDatasetTrain, SingleFaceDatasetValid
from lib import utils, checkpoint
import numpy as np


class ModelInterface(metaclass=abc.ABCMeta):
    """
    Base class for face GAN models. This base class can also be used 
    for neural network models with different purposes if some of concrete methods 
    are overrided appropriately. Exceptions will be raised when subclass is being 
    instantiated but abstract methods were not implemented. 
    """

    def __init__(self, args, gpu):
        """
        When overrided, super call is required.
        """
        self.args = args
        self.gpu = gpu
        self.dict = {}
        self.SetupModel()

    def SetupModel(self):
        self.args.isMaster = self.gpu == 0
        self.RandomGenerator = np.random.RandomState(42)
        self.set_networks()
        self.set_optimizers()

        if self.args.use_mGPU:
            self.set_multi_GPU()

        if self.args.load_ckpt:
            self.load_checkpoint()

        self.set_dataset()
        self.set_data_iterator()
        # self.set_validation()
        self.set_loss_collector()

        if self.args.isMaster:
            print(f'Model {self.args.model_id} has successively created')

    def load_next_batch(self):
        """
        Load next batch of source image, target image, and boolean values that denote 
        if source and target are identical.
        """
        try:
            vectors = next(self.train_iterator)
            # imgs, segs, vis, sketches, high_vectors, low_vectors = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_dataloader)
            vectors = next(self.train_iterator)
            # imgs, segs, vis, sketches, high_vectors, low_vectors = next(self.train_iterator)
        vectors = vectors.to(self.gpu)
        # imgs, segs, vis, sketches, high_vectors, low_vectors = imgs.to(self.gpu), segs.to(self.gpu), vis.to(self.gpu), sketches.to(self.gpu), high_vectors.to(self.gpu), low_vectors.to(self.gpu)
        return vectors
        # return imgs, segs, vis, sketches, high_vectors, low_vectors

    def set_dataset(self):
        """
        Initialize dataset using the dataset paths specified in the command line arguments.
        """
        self.train_dataset = SingleFaceDatasetTrain(self.args.train_dataset_root_list, self.args.isMaster)
        if self.args.valid_dataset_root:
            self.valid_dataset = SingleFaceDatasetValid(self.args.valid_dataset_root, self.args.isMaster)

    def set_data_iterator(self):
        """
        Construct sampler according to number of GPUs it is utilizing.
        Using self.dataset and sampler, construct dataloader.
        Store Iterator from dataloader as a member variable.
        """
        sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.args.use_mGPU else None
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_per_gpu, pin_memory=True, sampler=sampler, num_workers=8, drop_last=True)
        self.train_iterator = iter(self.train_dataloader)

    def set_validation(self):
        """
        Predefine test images only if args.valid_dataset_root is specified.
        These images are anchored for checking the improvement of the model.
        """
        if self.args.use_validation:
            self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.args.batch_per_gpu, num_workers=8, drop_last=True)
            I_source1, I_source2 = next(iter(self.valid_dataloader))
            self.valid_source1, self.valid_source2 = I_source1.to(self.gpu), I_source2.to(self.gpu)

    @abc.abstractmethod
    def set_networks(self):
        """
        Construct networks, send it to GPU, and set training mode.
        Networks should be assigned to member variables.

        eg. self.D = Discriminator(input_nc=3).cuda(self.gpu).train() 
        """
        pass

    def set_multi_GPU(self):
        utils.setup_ddp(self.gpu, self.args.gpu_num)

        # Data parallelism is required to use multi-GPU
        self.E = torch.nn.parallel.DistributedDataParallel(self.E, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True).module
        # self.D = torch.nn.parallel.DistributedDataParallel(self.D, device_ids=[self.gpu]).module


    def save_checkpoint(self, global_step):
        """
        Save model and optimizer parameters.
        """
        checkpoint.save_checkpoint(self.args, self.E, self.opt_E, name='E', global_step=global_step)
        # checkpoint.save_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=self.global_step, w_avg=self.w_avg)
        
        if self.args.isMaster:
            print(f"\nCheckpoints are succesively saved in {self.args.save_root}/{self.args.run_id}/ckpt/\n")
    
    def load_checkpoint(self):
        """
        Load pretrained parameters from checkpoint to the initialized models.
        """

        self.global_step = \
        checkpoint.load_checkpoint(self.args, self.E, self.opt_E, "E")
        # checkpoint.load_checkpoint(self.args, self.D, self.opt_D, "D")

        if self.args.isMaster:
            print(f"Pretrained parameters are succesively loaded from {self.args.save_root}/{self.args.ckpt_id}/ckpt/")

    def set_optimizers(self):
        if self.args.optimizer == "Adam":
            self.opt_E = torch.optim.Adam(self.E.parameters(), lr=self.args.lr_E, betas=self.args.betas)
        # if self.args.optimizer == "Ranger":
            # self.opt_E = Ranger(self.E.parameters(), lr=self.args.lr_E, betas=self.args.betas)
        # self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.args.lr_D, betas=(self.args.beta1, self.args.beta2))

    @abc.abstractmethod
    def set_loss_collector(self):
        """
        Set self.loss_collector as an implementation of lib.loss.LossInterface.
        """
        pass

    @property
    @abc.abstractmethod
    def loss_collector(self):
        """
        loss_collector should be an implementation of lib.loss.LossInterface.
        This property should be assigned in self.set_loss_collector.
        """
        pass

    @abc.abstractmethod
    def go_step(self):
        """
        Implement a single iteration of training. This will be called repeatedly in a loop. 
        This method should return list of images that was created during training.
        Returned images are passed to self.save_image and self.save_image is called in the 
        training loop preiodically.
        """
        pass

    @abc.abstractmethod
    def do_validation(self):
        """
        Test the model using a predefined valid set.
        This method includes util.save_image and returns nothing.
        """
        pass