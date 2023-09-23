from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="fast_forward",
    ext_modules=[
        CppExtension("fast_forward", ["./ops/fast_forward.cpp"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
