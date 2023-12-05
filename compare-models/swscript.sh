#!/bin/sh
#echo "wandb: Creating sweep with ID: qfathf2d" | awk '/ID:/ {print $6}'
#wandb  sweep sweep_config.yaml --project mvnn   
export tmp=$(wandb  sweep sweep_config.yaml --project MVNN-Runs 2>&1 | awk '/ID:/ {print $6}')
echo "sweep ID is: $tmp " 

#wandb agent --count 200 mvnn-ma/mvnn/$tmp

for i in {0..10}
do 
    wandb agent --count 200 mvnn-ma/mvnn/$tmp
done    

#export SWEEP_ID=$(awk '/ID:/ {print $6}' tmp )
#echo $SWEEP_ID
#input=tmp
#while read -r line 
#do 
#    echo $line
    #export SWEEP_ID=$($line | awk '/ID:/ {print $6}' )
#done < "$line"   
#echo $SWEEP_ID
