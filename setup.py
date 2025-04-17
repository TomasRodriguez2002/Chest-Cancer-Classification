import setuptools
import os
from dotenv import load_dotenv
load_dotenv()

repo_name = os.getenv("REPO_NAME")
author_user_name = os.getenv("AUTHOR_USER_NAME")
src_repo = os.getenv("SRC_REPO")
author_email = os.getenv("AUTHOR_EMAIL")

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"


setuptools.setup(
    name=src_repo,
    version=__version__,
    author=author_user_name,
    author_email=author_email,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{author_user_name}/{repo_name}",
    project_urls={
        "Bug Tracker": f"https://github.com/{author_user_name}/{repo_name}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)