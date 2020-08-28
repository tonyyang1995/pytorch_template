from .BaseOptions import BaseOptions 

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        self.isTrain = False
        return parser 