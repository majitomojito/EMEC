

function indexOfMax(arr) {
	// returns the index the element with max value of an array
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}

//https://stackoverflow.com/questions/11301438/return-index-of-greatest-value-in-an-array/11301464

function indexOfMin(arr) {
	// returns the index of element with min value of an array
    if (arr.length === 0) {
        return 0;
    }

    var min = arr[0];
    var minIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] < min) {
            minIndex = i;
            min = arr[i];
        }
    }

    return minIndex;
}
