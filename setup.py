import setuptools
setuptools.setup(
    name="ogre",
    version="0.0.1",
    author="Sander Borgmans",
    author_email="sander.borgmans@ugent.be",
    description="Optimal Grid Refinement tool",
    url="https://github.com/SanderBorgmans/OGRe",
    packages=setuptools.find_packages(),
    scripts=['scripts/ogre_input.py', 'scripts/ogre_sim.py', 'scripts/ogre_post.py', 'scripts/ogre_compress_iteration.py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
