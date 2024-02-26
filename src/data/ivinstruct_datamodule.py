from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from transformers import AutoTokenizer, AutoProcessor
from timm.data import create_transform

from .components.ivinstruct_dataset import IVINSTRUCT
from src.gadgets.transforms import RandomResizedCropVideo, ToTHWC, ToUint8, ToTensorVideo, NormalizeVideo, ResizeVideo

def _convert_to_rgb(image):
    return image.convert('RGB')

class IVINSTRUCTDataModule(LightningDataModule):
    """`LightningDataModule` for the VideoInstruct dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        text_dir: str = "data/",
        image_dir: str = "data/",
        video_dir: str = "data/",
        of_dir: str = "data/",
        nframe: int = 4,
        processor_name: str = '',
        sampler_processor_name: str = '',
        target_size: Optional[int] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes.
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        
        Or you can test if your dataset is available
        """
        pass
        

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # set data processor
        # # Transform Image
        self.imagetransforms = Compose([
            Resize((self.hparams.target_size, self.hparams.target_size), interpolation=InterpolationMode.BICUBIC),
            # _convert_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        # # Transform video
        self.videotransforms = Compose([
                ResizeVideo(self.hparams.target_size),
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                ToTensorVideo(),  # C, T, H, W
                NormalizeVideo((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # C, T, H, W
            
        ])
        
        # self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(self.hparams.processor_name)
        # print(self.hparams.processor_name)
        # if 'vicuna' in self.hparams.processor_name:
        #     # raise ValueError("debug")
        #     # print('test')
        #     self.processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #     self.processor.tokenizer.add_special_tokens({'bos_token': '</s>'})
        #     self.processor.tokenizer.add_special_tokens({'eos_token': '</s>'})
        #     self.processor.tokenizer.add_special_tokens({'unk_token': '</s>'}) 
        self.sampler_processor = AutoTokenizer.from_pretrained(self.hparams.sampler_processor_name)

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = IVINSTRUCT(self.hparams.text_dir, self.hparams.image_dir, self.hparams.video_dir, self.hparams.of_dir, nframe=self.hparams.nframe, split='train', processor=self.processor, sampler_processor=self.sampler_processor, video_transform=self.videotransforms, image_transform=self.imagetransforms)
            self.data_val = IVINSTRUCT(self.hparams.text_dir, self.hparams.image_dir, self.hparams.video_dir, self.hparams.of_dir, nframe=self.hparams.nframe, split='val', processor=self.processor, sampler_processor=self.sampler_processor, video_transform=self.videotransforms, image_transform=self.imagetransforms)
            self.data_test = IVINSTRUCT(self.hparams.text_dir, self.hparams.image_dir, self.hparams.video_dir, self.hparams.of_dir, nframe=self.hparams.nframe, split='test', processor=self.processor, sampler_processor=self.sampler_processor, video_transform=self.videotransforms, image_transform=self.imagetransforms)
            

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.data_train.collate,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.data_train.collate,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.data_train.collate,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass



if __name__ == "__main__":
    _ = IVINSTRUCTDataModule()
