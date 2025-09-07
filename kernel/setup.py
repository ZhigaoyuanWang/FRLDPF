from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

gcn_fusion_module = CppExtension(
    name='gcn_fusion',
    sources=['gcn_fusion.cpp']
)

# attention_fusion_module = CppExtension(
#     name='attention_fusion',
#     sources=['attention_fusion.cpp']
# )

setup(
    name='custom_ops',
    ext_modules=[
        gcn_fusion_module,
        # attention_fusion_module
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)