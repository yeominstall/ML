
def indexof(arr, item):
	for i in arr:
		print arr[i]
		if arr[i] == item:
			return i
	return -1
