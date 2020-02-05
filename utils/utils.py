import os
import shutil
import stat
from pathlib import Path

import git
import giturlparse


def delete_dir(dir_path):
    shutil.rmtree(dir_path, onerror=on_delete_error)


def on_delete_error(action, name, exc):
    os.chmod(name, stat.S_IWRITE)
    os.remove(name)


def empty_dir(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, onerror=on_delete_error)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def clone_by_url(repo_url):
    p = giturlparse.parse(repo_url)
    print(f'Fetching github repository {p.owner}/{p.repo}...')
    dir_path_str = f'./repos/{p.owner}/{p.repo}'
    dir_path = Path(dir_path_str)
    if dir_path.exists() and dir_path.is_dir():
        print(f'Repository {p.owner}/{p.repo} already exists! Removing old version...')
        delete_dir(dir_path)
    git.Repo.clone_from(repo_url, dir_path_str)
    print('OK')


def clone_all_from_user(username):
    os.system(
        f"python {os.path.join('githubcloner', 'githubcloner.py')} "
        f"--user {username} -o ./repos/ --prefix-mode directory")


def get_repo_list(username):
    print(f'Fetching github repositories of user \'{username}\'')
    output = os.popen(
        f"python {os.path.join('githubcloner', 'githubcloner.py')} "
        f"--user {username} --echo-urls").read()
    url_list = output.splitlines()
    return url_list
