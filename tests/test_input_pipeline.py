from tpu_data_provider import BaseInputPipeline, RandomSizedCocoDataProvider

class TestInputPipeline:

    def test_sample_load_train_loader(self):
        pipeline = BaseInputPipeline()
        train_loader = pipeline.load_train_loader()
        for i, data in enumerate(train_loader):
            inputs, labels = data

        assert len(inputs.shape) == 4
        assert inputs.shape[1] == 3

        # args.image_size = '128,160,192,224'

    def test_distributed_train_loader(self):
        # copied from once-for-all/ofa/imagenet_classification/elastic_nn/training/progressive_shrinking.py
        data_provider = RandomSizedCocoDataProvider()
        for i, (images, labels) in enumerate(data_provider.train):
            assert len(images.shape) == 4
            break

    def test_active_img_size(self):

        lengths = set()
        img_size = [128, 160, 192, 224]
        data_provider = RandomSizedCocoDataProvider(image_size=img_size)
        for i, (images, labels) in enumerate(data_provider.train):
            lengths.add(images.shape[2])
            if i == 10:
                assert len(lengths) > 2  # 랜덤하게 잘 나왔다는 걸 어떻게 보여줄까...

    def test_load_train_loader(self):
        pass

    def test_load_test_loader(self):
        pass

    def test_img_in_tpu(self):
        pass
