
#!/bin/bash

# Global Variables
PYTHON_FILE="assim_LETKF_cluster.py"
LINE_NUMBER_R=257
LINE_NUMBER_SEUIL=258
SHELL_SCRIPT="assim.sh"

# Function to update the Python file with the new R value
update_python_file_r_normal() {
    local r_value=$1
    local tmp_file
    tmp_file=$(mktemp) || { printf "Failed to create temp file\n" >&2; return 1; }

    if ! sed "${LINE_NUMBER_R}s/.$/${r_value}/" "$PYTHON_FILE" > "$tmp_file"; then
        printf "Failed to update the Python file for R value\n" >&2
        rm -f "$tmp_file"
        return 1
    fi

    if ! mv "$tmp_file" "$PYTHON_FILE"; then
        printf "Failed to move temp file to Python file\n" >&2
        rm -f "$tmp_file"
        return 1
    fi
}

update_python_file_r_debut() {
    local r_value=$1
    local tmp_file
    tmp_file=$(mktemp) || { printf "Failed to create temp file\n" >&2; return 1; }

    if ! sed "${LINE_NUMBER_R}s/..$/${r_value}/" "$PYTHON_FILE" > "$tmp_file"; then
        printf "Failed to update the Python file for R value\n" >&2
        rm -f "$tmp_file"
        return 1
    fi

    if ! mv "$tmp_file" "$PYTHON_FILE"; then
        printf "Failed to move temp file to Python file\n" >&2
        rm -f "$tmp_file"
        return 1
    fi
}

# Function to update the Python file with the new seuil value
update_python_file_seuil() {
    local seuil_value=$1
    local tmp_file
    tmp_file=$(mktemp) || { printf "Failed to create temp file\n" >&2; return 1; }

    if ! sed "${LINE_NUMBER_SEUIL}s/..$/${seuil_value}/" "$PYTHON_FILE" > "$tmp_file"; then
        printf "Failed to update the Python file for seuil value\n" >&2
        rm -f "$tmp_file"
        return 1
    fi

    if ! mv "$tmp_file" "$PYTHON_FILE"; then
        printf "Failed to move temp file to Python file\n" >&2
        rm -f "$tmp_file"
        return 1
    fi
}

# Function to run the shell script and wait for it to finish
run_shell_script() {
    if ! bash "$SHELL_SCRIPT"; then
        printf "Failed to execute the shell script\n" >&2
        return 1
    fi
}

# Main function to loop over the seuil values and R values, and execute the tasks
main() {
    local seuil r seuil_p
    

    #seuil_p=20
    #if ! update_python_file_seuil "$seuil_p"; then
            #printf "Error updating Python file for seuil=%d\n" "$seuil_p" >&2
            #continue
        #fi

        #for r in $(seq 7 10); do
            #if ! update_python_file_r "$r"; then
                #printf "Error updating Python file for R=%d\n" "$r" >&2
                #continue
            #fi

            #if ! run_shell_script; then
                #printf "Error running shell script for R=%d and seuil=%d\n" "$r" "$seuil_p" >&2
                #continue
            #fi
        #done


    for seuil in 10 20 30; do
        if ! update_python_file_seuil "$seuil"; then
            printf "Error updating Python file for seuil=%d\n" "$seuil" >&2
            continue
        fi
        r=2
        if ! update_python_file_r_debut "$r"; then
                printf "Error updating Python file for R=%d\n" "$r" >&2
                continue
            fi

        for r in 23; do
            if ! update_python_file_r_normal "$r"; then
                printf "Error updating Python file for R=%d\n" "$r" >&2
                continue
            fi

            if ! run_shell_script; then
                printf "Error running shell script for R=%d and seuil=%d\n" "$r" "$seuil" >&2
                continue
            fi
        done
    done
}

main "$@"