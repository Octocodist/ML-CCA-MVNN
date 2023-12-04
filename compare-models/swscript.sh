#echo "wandb: Creating sweep with ID: qfathf2d" | awk '/ID:/ {print $6}'
export tmp=$(wandb sweep sweep_config.yaml --project mvnn 2>&1 | awk '/ID:/ {print $6}')
echo "sweep ID is: $tmp " 
#export SWEEP_ID=$(awk '/ID:/ {print $6}' tmp )
#echo $SWEEP_ID
#input=tmp
#while read -r line 
#do 
#    echo $line
    #export SWEEP_ID=$($line | awk '/ID:/ {print $6}' )
#done < "$line"   
#echo $SWEEP_ID
#rm tmp
wandb agent --count 50 $tmp


