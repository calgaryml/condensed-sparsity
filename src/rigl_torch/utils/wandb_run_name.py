from rigl_torch.exceptions import WandbRunNameException  

class WandbRunName():
    
    def __init__(self, name: str):
        self.name = name
        self._verify_name()
    
    def _verify_name(self):
        if " " in self.name:
            raise WandbRunNameException(
                message="No spaces allowed in name",
                name=self.name
            )
        if len(self.name) > 128:
            raise WandbRunNameException(
                message="Name must be <= 128 chars",
                name=self.name
            )
        