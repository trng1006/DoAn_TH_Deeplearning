#!/bin/bash
cd "/Users/tranbaonguyen/Documents/DoAn_TH_DeepLearning/DoAn_TH_Deeplearning"
git init
git remote add origin https://github.com/trng1006/DoAn_TH_Deeplearning.git || git remote set-url origin https://github.com/trng1006/DoAn_TH_Deeplearning.git
git add .
git commit -m "Upload project DoAn_TH_Deeplearning"
git branch -M main
git push -u origin main
