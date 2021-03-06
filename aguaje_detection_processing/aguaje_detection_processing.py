# -*- coding: utf-8 -*-

"""
/***************************************************************************
 AguajeDetection
                                 A QGIS plugin
 This plugin adds an algorithm to detect Aguaje in a raster layer.
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2019-12-12
        copyright            : (C) 2019 by Susan Palacios, University of Brescia
        email                : s.palaciossalcedo@studenti.unibs.it
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

__author__ = 'Susan Palacios, University of Brescia'
__date__ = '2019-12-12'
__copyright__ = '(C) 2019 by Susan Palacios, University of Brescia'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

import os
import sys
import inspect

from qgis.PyQt.QtWidgets import QAction#icon
from qgis.PyQt.QtGui import QIcon#icon

from qgis.core import QgsProcessingAlgorithm, QgsApplication
import processing #icon
from .aguaje_detection_processing_provider import AguajeDetectionProvider

cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]

if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)


class AguajeDetectionPlugin(object):

    def __init__(self, iface):
        self.provider = None
        self.iface = iface

    def initProcessing(self):
        """Init Processing provider for QGIS >= 3.8."""
        self.provider = AguajeDetectionProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()
        icon = os.path.join(os.path.join(cmd_folder, 'logo.png'))#icon
        self.action = QAction(
            QIcon(icon),
            u"Detection algorithm", self.iface.mainWindow())#icon
        self.action.triggered.connect(self.run)#icon
        self.iface.addPluginToMenu(u"&Aguaje", self.action)#icon
        self.iface.addToolBarIcon(self.action)#icon

    def unload(self):
        QgsApplication.processingRegistry().removeProvider(self.provider)
        self.iface.removePluginMenu(u"&Aguaje", self.action)#icon
        self.iface.removeToolBarIcon(self.action)#icon

    def run(self):#icon
        processing.execAlgorithmDialog("Aguaje:Detection algorithm")#icon
