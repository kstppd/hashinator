#!/bin/bash

make allclean >/dev/null 2>&1
clear
echo "********** Running Hashinator Tests ***************"
tests=($(cat Makefile | grep "OBJ=" | awk -F " "  '{for(i=2;i<=NF;i++){printf "%s\n", $i};}'))
all_tests_passed=true
for i in "${tests[@]}"
do
	echo Building test: $i
   make $i >/dev/null 2>&1
   if [ $? -eq 1 ]; then
     echo Compilation of  $i failed!
     exit 1
   fi
   file=$(find . -executable -type f \( ! -iname "*sh" \) -printf "%p\n")
   echo Build file=  $file
	echo Running test: $i
   $file >/dev/null 2>&1
   result=$?
   if [ $result -eq 0 ]; then
     echo echo Test $file passed!
   else
     echo echo Test $file failed!
     all_tests_passed=false
   fi
   make allclean >/dev/null 2>&1
   echo " "
done

make allclean >/dev/null 2>&1
if [ $all_tests_passed ]; then 
   echo "ALL TESTS PASSED"
fi
