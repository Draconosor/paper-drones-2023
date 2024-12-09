#!/usr/bin/env python

# Copyright (c) 2017 NEOS-Server
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Python source code - Python XML-RPC client for NEOS Server
"""

import os
import sys
import xmlrpc.client as xmlrpclib
from time import sleep
import pandas as pd

def get_subfolders(path):
    subfolders = [f.name for f in os.scandir(path) if f.is_dir() and f.name != 'results']
    return subfolders

def create_folder(directory):
    # Check if the directory already exists
    if os.path.exists(directory):
        # If it exists, remove it
        try:
            os.rmdir(directory)
            print(f"Existing directory '{directory}' removed.")
        except OSError as e:
            print(f"Error removing directory '{directory}': {e}")
            return

    # Create the new directory
    try:
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    except OSError as e:
        print(f"Error creating directory '{directory}': {e}")


class NEOSJob:
    def __init__(self, username:str, password:str, instance:str, objective: str):
        self.instance_id = instance + objective
        instance = os.path.join(r'instances',instance)
        self.neos = xmlrpclib.ServerProxy("https://neos-server.org:3333")
        self.username = username
        self.password = password
        self.objective = objective
        self.action = os.path.join(instance, f'NEOS INSTRUCTION {objective}.xml')
        self.instance = instance
        self.job_id = None
        self.job_password = None
        
    def set_job_credentials(self, job_id:str, job_password:str):
        self.job_id = job_id
        self.job_password = job_password
    def retrieve_results(self, path = None):
        if self.job_id and self.job_password:
            msg = self.neos.getFinalResults(self.job_id, self.job_password)
            files = self.neos.getOutputFile(self.job_id, self.job_password, 'solver-output.zip')
            if path:
                with open(os.path.join(path, f'NEOS RESULTS {self.instance_id} {self.objective}.lst'), 'w') as file:
                    file.write(msg.data.decode())      
                with open(os.path.join(path, f'NEOS RESULTS {self.instance_id} {self.objective}.zip'), 'wb') as file:
                    file.write(files.data)     
            else:
                with open(os.path.join(self.instance, f'NEOS RESULTS {self.instance_id} {self.objective}.lst'), 'w') as file:
                    file.write(msg.data.decode())      
                with open(os.path.join(path, f'NEOS RESULTS {self.instance_id} {self.objective}.zip'), 'wb') as file:
                    file.write(files.data)    
            print(f"Data saved to {os.path.join(self.instance, f'NEOS RESULTS {self.instance} {self.objective}.lst')}")


def send_to_NEOS(job: NEOSJob):
    neos = job.neos

    alive = neos.ping()
    if alive != "NeosServer is alive\n":
        sys.stderr.write("Could not make connection to NEOS Server\n")
        sys.exit(1)

    if job.action == "queue":
        msg = neos.printQueue()
        sys.stdout.write(msg)
    else:
        xml = ""
        try:
            xmlfile = open(job.action, "r")
            buffer = 1
            while buffer:
                buffer = xmlfile.read()
                xml += buffer
            xmlfile.close()
        except IOError as e:
            sys.stderr.write("I/O error(%d): %s\n" % (e.errno, e.strerror))
            sys.exit(1)
        if job.username and job.password:
            (jobNumber, password) = neos.authenticatedSubmitJob(xml, job.username, job.password)
        else:
            (jobNumber, password) = neos.submitJob(xml)
        if jobNumber != 0:
            job.set_job_credentials(job_id=jobNumber, job_password=password)
            return job
        else:
            raise Exception('Job not scheduled properly')
def main():       
    instances = [i for i in os.listdir('instances') if i.startswith('C')]
    jobs_df = pd.DataFrame(columns = ['instance', 'objective','jobId', 'password'])
    c = 0
    for ins in instances:
        if c >= 15:
            sleep(7260)
            c = 0

        steps = [i[-12:][:8] for i in os.listdir(os.path.join('instances', ins)) if i.endswith('pct.gms')]
        for name in steps:
            myjob = NEOSJob(os.environ.get('NEOS_USERNAME'), os.environ.get('NEOS_PASSWORD'), ins,name)
            uploaded_job = send_to_NEOS(myjob)
            mydata = pd.DataFrame({'instance': [uploaded_job.instance_id], 'objective': [uploaded_job.objective],'jobId':[uploaded_job.job_id], 'password':[uploaded_job.job_password]})
            jobs_df = pd.concat([jobs_df, mydata])
        c += 5
    with pd.ExcelWriter('jobs_pct_df.xlsx', engine='xlsxwriter') as writer:
        jobs_df.to_excel(writer, sheet_name = 'Jobs', index = False)

def retrieve_batch_results():
    jobs_df = pd.read_excel('jobs_pct_df.xlsx')
    ## MAKE RESULTS DIR
    create_folder(os.path.join(r'instances', 'results pcts'))
    for index, row in jobs_df.iterrows():
        temp_job = NEOSJob(os.environ.get('NEOS_USERNAME'), os.environ.get('NEOS_PASSWORD'), row.instance[:-8], row.objective)
        temp_job.set_job_credentials(row.jobId, row.password)
        temp_job.retrieve_results(path = os.path.join(r'instances', 'results pcts'))


def extract_and_print_substr(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) >= 55:
            line_55 = lines[54]  # Line numbering starts from 0
            if len(line_55) >= 44:  # Ensure the line is long enough
                between_30_and_minus_14 = line_55[29:-14]
                print(f"File: {file_path}, {between_30_and_minus_14}")

def iterate_over_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.lst'):
                file_path = os.path.join(root, file)
                extract_and_print_substr(file_path)

retrieve_batch_results()
# Replace 'your_path_here' with the actual path you want to iterate over
#iterate_over_files(r'C:\Users\Carlos\OneDrive - Universidad de la Sabana\MGOP\DRONES 2023 2\paper-drones-2023\instances\results pcts')
