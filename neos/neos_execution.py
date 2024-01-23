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
import time


class NEOSJob:
    def __init__(self, username:str, password:str, instance:str):
        self.neos = xmlrpclib.ServerProxy("https://neos-server.org:3333")
        self.username = username
        self.password = password
        self.action = os.path.join(instance, 'NEOS INSTRUCTION.xml')
        self.instance = instance
        self.job_id = None
        self.job_password = None
        
    def set_job_credentials(self, job_id:str, job_password:str):
        self.job_id = job_id
        self.job_password = job_password
    def retrieve_results(self):
        if self.job_id and self.job_password:
            msg = self.neos.getFinalResults(self.job_id, self.job_password)
            with open(os.path.join(self.instance, 'NEOS RESULTS.lst'), 'w') as file:
                file.write(msg.data.decode())      
            print(f"Data saved to {os.path.join(self.instance, 'NEOS RESULTS.lst')}")


def main(job: NEOSJob):
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


myjob = NEOSJob(os.environ.get('NEOS_USERNAME'), os.environ.get('NEOS_PASSWORD'), instance=r'instances/C10P5T5D10')

updated_job = main(myjob)
time.sleep(45)
updated_job.retrieve_results()