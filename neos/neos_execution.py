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
import pandas as pd
from time import sleep

def get_subfolders(path):
    subfolders = [f.name for f in os.scandir(path) if f.is_dir() and f.name != 'results']
    return subfolders


class NEOSJob:
    def __init__(self, username:str, password:str, instance:str, objective: str):
        self.instance_id = instance
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
            print(f"Data saved to {os.path.join(self.instance, f'NEOS RESULTS {self.instance_id} {self.objective}.lst')}")


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
            print(xml)
            raise Exception('Job not scheduled properly')
        

objectives = ('z1',) 

def send_batch_to_neos():
    jobs_df = pd.DataFrame(columns = ['instance', 'objective','jobId', 'password'])
    for instance in [x for x in get_subfolders(r'instances') if 'C' in x]:
        for obj in objectives:
            myjob = NEOSJob(os.environ.get('NEOS_USERNAME'), os.environ.get('NEOS_PASSWORD'), instance= instance, objective = obj)
            uploaded_job = send_to_NEOS(myjob)
            mydata = pd.DataFrame({'instance': [uploaded_job.instance], 'objective': [uploaded_job.objective],'jobId':[uploaded_job.job_id], 'password':[uploaded_job.job_password]})
            jobs_df = pd.concat([jobs_df, mydata])
            sleep(1)
    with pd.ExcelWriter('jobs_df.xlsx', engine='xlsxwriter') as writer:
        jobs_df.to_excel(writer, sheet_name = 'Jobs', index = 'False')
    print('Success!')

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

def retrieve_batch_results():
    jobs_df = pd.read_excel('jobs_df.xlsx')
    ## MAKE RESULTS DIR
    create_folder(os.path.join(r'instances', 'results'))
    for index, row in jobs_df.iterrows():
        temp_job = NEOSJob(os.environ.get('NEOS_USERNAME'), os.environ.get('NEOS_PASSWORD'), row.instance, row.objective)
        temp_job.set_job_credentials(row.jobId, row.password)
        temp_job.retrieve_results(path = os.path.join(r'instances', 'results'))

retrieve_batch_results()