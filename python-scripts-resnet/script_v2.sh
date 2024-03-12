
while true; do
    # Get the free memory percentage using nvidia-smi and extract the value
    free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}')

    # Set the threshold percentage
    threshold=0
    #echo "Free memory: $free_memory MB"
    # Compare the free memory percentage with the threshold
    if (( free_memory > threshold )); then
        echo "GPU memory is free enough. Running Python script..."
        python -u Main_v2.py

        break
    else
        #echo "Free memory: $free_memory MB"
        current_date_time=$(date "+%A, %B %d, %Y %T %Z")
        echo "$current_date_time: GPU memory is not free enough"
    fi

    # Wait for some time before checking again (e.g., every 5 minutes)
    sleep 300  # 300 seconds = 5 minutes
done
