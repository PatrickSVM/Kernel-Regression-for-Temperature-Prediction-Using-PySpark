#!/bin/bash
SBATCH --time=20:00
#SBATCH --nodes=2
#SBATCH --exclusive

module add spark/2.4.3-hadoop-2.7-nsc1

# Cleanup and start from scratch
rm -rf spark

echo "START AT: $(date)"

hadoop_setup

echo "Prepare output and input directories and files..."
# The following command will make folders on your home folder on HDFS, the input and output folders should be corresponding to the parameter you give to textFile and saveAsTextFile functions in the code
hadoop fs -mkdir -p "BDA" "BDA/input"
hadoop fs -test -d "BDA/output"
if [ "$?" == "0" ]; then
    hadoop fs -rm -r "BDA/output"
fi

#hadoop fs -copyFromLocal ./input_data/temperature-readings-small.csv "BDA/input/"
# Remove the comment when you need specifc file below
hadoop fs -copyFromLocal ../input_data/temperature-readings.csv "BDA/input/"
#hadoop fs -copyFromLocal ../input_data/precipitation-readings.csv "BDA/input/"
hadoop fs -copyFromLocal ../input_data/stations.csv "BDA/input/"
#hadoop fs -copyFromLocal ../input_data/stations-Ostergotland.csv "BDA/input/"

# Run your program
echo "Running Your program..."
exec 5>&1
APPLICATION_ID=$(spark-submit --conf spark.eventLog.enabled=true --deploy-mode cluster --master yarn --num-executors 9 --driver-memory 2g --executor-memory 2g --executor-cores 4 implementation.py 2>&1 | tee >(cat - >&5) | awk '!found && /INFO.*Yarn.*Submitted application/ {tmp=gensub(/^.*Submitted application (.*)$/,"\\1","g");print tmp; found=1}')

echo "================= FINAL OUTPUT =========================================="
hadoop fs -cat "BDA/output"/*
echo "========================================================================="
echo "Applicaton id: $APPLICATION_ID"
echo "========================================================================="
echo "=                              stderr                                   ="
echo "========================================================================="
yarn logs -applicationId "$APPLICATION_ID" | awk -F: '/^LogType/ {if($2=="stderr") {output=1} else {output=0}} output==1 {print}'
echo "========================================================================="
echo "=                              result                                   ="
echo "========================================================================="
yarn logs -applicationId "$APPLICATION_ID" | awk -F: '/^LogType/ {if($2=="stdout") {output=1} else {output=0}} output==1 {print}' | grep -v "WARN\|INFO"

rm -rf output
hadoop fs -copyToLocal 'BDA/output' ./
hadoop_stop

echo "END AT: $(date)"
