# -*- coding: utf-8 -*-
# @Author   : Hyo Kong
# @Email    : hyokong@stu.xjtu.edu.cn
# @Project  : HiMask
# @Time     : 20/5/21 4:12 PM
# @File     : _initPath.py
# @Function :

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

def addPath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

thisDir = os.path.dirname(__file__)
libPath = os.path.join(thisDir, '..', 'lib')
addPath(libPath)