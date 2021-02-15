import subprocess

if __name__ == '__main__':
    N_network_try = 2000
    for frame in range(N_network_try):
        # subprocess.run(["ls", "-l"])
        subprocess.run(["python", "test_build_active_subnet.py"])


# # load
# with open('data.pickle', 'rb') as f:
#     data = pickle.load(f)