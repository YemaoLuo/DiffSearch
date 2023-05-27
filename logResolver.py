import os
import subprocess

import torch
from tqdm import tqdm

import sentence2Vector

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def resolve_log(path='.'):
    all_log = []
    scan_files = ['py', 'java', 'js']
    output = subprocess.check_output(['git', 'log', '--pretty=%H %an %ae -%ad'], cwd=path)
    commit_ids = []
    commit_authors = []
    commit_times = []
    commit_info = output.decode('utf-8').split('\n')
    for info in commit_info:
        try:
            commit_ids.append(info.split(' ')[0])
            commit_authors.append(info.split(' ')[1] + ' ' + info.split(' ')[2])
            commit_times.append(info.split('-')[-1])
        except IndexError:
            continue
    pbar = tqdm(total=len(commit_ids) - 1, desc='Processing', unit='items', position=0, leave=True)
    for i in range(len(commit_ids) - 1):
        pbar.update(1)
        if len(commit_ids[i + 1]) == 0:
            break
        output = subprocess.check_output(['git', 'show', commit_ids[i + 1], '--format=%B'], cwd=path)
        try:
            current_log = {'id': commit_ids[i + 1], 'author': commit_authors[i + 1], 'time': commit_times[i + 1]
                , 'message': output.decode('utf-8').split('\n')[0]}
        except UnicodeDecodeError:
            continue
        if current_log['message'].__contains__('Merge'):
            continue
        torch.save(sentence2Vector.get_embedding(current_log['message']),
                   'log/tensor/' + commit_ids[i + 1])
        diff_files = []
        diff = ''
        for line in output.decode('utf-8').split('\n'):
            file_map = {}
            if line.__contains__('diff --git'):
                file_map['difference'] = diff
                diff = ''
                diff_files.append(file_map)
                file_map['file_name'] = line.split('/')[-1]
            else:
                diff += line + '\n'
        current_log['diff'] = diff_files
        all_log.append(current_log)

        log_file = open('log/' + current_log['id'] + '.log', 'w')
        log_file.write(current_log['id'] + '\n')
        log_file.write('=' * 50 + '\n')
        log_file.write(current_log['message'] + '\n')
        log_file.write('=' * 50 + '\n')
        log_file.write(current_log['author'] + '\n')
        log_file.write('=' * 50 + '\n')
        log_file.write(current_log['time'] + '\n')
        log_file.write('=' * 50 + '\n')
        for diff_file in current_log['diff']:
            if diff_file['file_name'].split('.')[-1] in scan_files:
                log_file.write(diff_file['file_name'] + '\n')
                log_file.write(diff_file['difference'] + '\n')
                log_file.write('=' * 50 + '\n')
        log_file.close()
    return all_log


if __name__ == '__main__':
    os.mkdir('log')
    os.mkdir('log/tensor')
    resolve_log('./tomcat')
    # resolve_log()
