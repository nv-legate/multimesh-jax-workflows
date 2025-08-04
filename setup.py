from setuptools import find_packages, setup

setup(
    name="MultiMesh Jax Workflows",
    description="Various tasks for building and testing MultiMesh for Jax",
    url="TBD",
    author="NVIDIA Corporation",
    license="Closed source",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=find_packages(include=["multimesh_jax_workflows"]),
    entry_points={
        "console_scripts": [
            "run_multimesh_testsuite=multimesh_jax_workflows:run_multimesh_testsuite.entrypoint",  # noqa: E501
            "run_maxtext=multimesh_jax_workflows:run_maxtext.entrypoint",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
