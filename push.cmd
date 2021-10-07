@echo off
eval `ssh-agent -s`
ssh-add ~/.ssh/id_rsa_cv
git push
cmd /k


