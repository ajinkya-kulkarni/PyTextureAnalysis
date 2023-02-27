#!/usr/bin/env python3
# encoding: utf-8
#
# Copyright (C) 2022 Max Planck Institute for Multidisclplinary Sciences
# Copyright (C) 2022 University Medical Center Goettingen
# Copyright (C) 2022 Ajinkya Kulkarni <ajinkya.kulkarni@mpinat.mpg.de>
# Copyright (C) 2022 Bharti Arora <bharti.arora@mpinat.mpg.de>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

########################################################################################

# This file contains the parameters necessary to run this package in a standalone manner

########################################################################################

# Define filename to be analyzed:

filename = 'TestImage1.tif'

########################################################################################

# Analysis parameters

FilterKey = 1

LocalSigmaKey = 10

BinarizationKey = 20

LocalDensityKey = 10

ThresholdValueKey = 20

SpacingKey = 20

ScaleKey = 40

########################################################################################

# Plotting parameters

FIGSIZE = (15, 8)
PAD = 5
FONTSIZE_TITLE = 14
DPI = 500

aspect = 30
pad_fraction = 0.5

########################################################################################