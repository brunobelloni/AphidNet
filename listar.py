# -*- coding:utf-8 -*-
import glob
import os

diretorio = 'flower_photos/'
pastas = []


def listar(diretorio):
    if os.path.isdir(diretorio):
        os.chdir(diretorio)
        for arquivo in glob.glob("*"):
            if os.path.isdir(diretorio + arquivo):
                listar(diretorio + arquivo + '/')
            else:
                print('arquivo: ' + diretorio + arquivo)
    else:
        print('arquivo: ' + diretorio)


print(listar(diretorio))
