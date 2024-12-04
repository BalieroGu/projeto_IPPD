#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef struct {
    float x;
    float y;
} Point;

typedef struct {
    char label;
    int length;
    Point *points;
} Group;

void on_error() {
    printf("Invalid input file.\n");
    exit(1);
}

int parse_number_of_groups() {
    int n;
    if (scanf(" n_groups=%d ", &n) != 1) on_error();
    return n;
}

Point parse_point() {
    float x, y;
    if (scanf(" (%f ,%f) ", &x, &y) != 2) on_error();
    return (Point){x, y};
}

Group parse_next_group() {
    char label;
    int length;

    if (scanf(" label=%c ", &label) != 1) on_error();
    if (scanf(" length=%d ", &length) != 1) on_error();

    Group group;
    group.label = label;
    group.length = length;
    group.points = (Point *)malloc(sizeof(Point) * length);

    for (int i = 0; i < length; i++) {
        group.points[i] = parse_point();
    }
    return group;
}

int parse_k() {
    int k;
    if (scanf(" k=%d ", &k) != 1) on_error();
    return k;
}

float euclidean_distance_no_sqrt(Point a, Point b) {
    return ((b.x - a.x) * (b.x - a.x)) + ((b.y - a.y) * (b.y - a.y));
}

void insert_into_heap(float *distances, char *labels, float distance, char label, int k) {
    if (distance >= distances[0]) return; // Heap root is largest; skip if larger
    distances[0] = distance;
    labels[0] = label;

    // Restore heap property
    int i = 0;
    while (1) {
        int left = 2 * i + 1, right = 2 * i + 2, smallest = i;
        if (left < k && distances[left] > distances[smallest]) smallest = left;
        if (right < k && distances[right] > distances[smallest]) smallest = right;
        if (smallest == i) break;
        float temp_d = distances[i];
        distances[i] = distances[smallest];
        distances[smallest] = temp_d;
        char temp_l = labels[i];
        labels[i] = labels[smallest];
        labels[smallest] = temp_l;
        i = smallest;
    }
}

char knn(int n_groups, Group *groups, int k, Point to_evaluate) {
    char *labels = (char *)malloc(sizeof(char) * k);
    float *distances = (float *)malloc(sizeof(float) * k);
    for (int i = 0; i < k; i++) distances[i] = INFINITY;

    #pragma omp parallel
    {
        float *local_distances = (float *)malloc(sizeof(float) * k);
        char *local_labels = (char *)malloc(sizeof(char) * k);
        for (int i = 0; i < k; i++) local_distances[i] = INFINITY;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_groups; i++) {
            Group g = groups[i];
            for (int j = 0; j < g.length; j++) {
                float d = euclidean_distance_no_sqrt(to_evaluate, g.points[j]);
                insert_into_heap(local_distances, local_labels, d, g.label, k);
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < k; i++) {
                insert_into_heap(distances, labels, local_distances[i], local_labels[i], k);
            }
        }

        free(local_distances);
        free(local_labels);
    }

    int *frequency = (int *)calloc(256, sizeof(int));
    for (int i = 0; i < k; i++) frequency[(unsigned char)labels[i]]++;

    char most_frequent = labels[0];
    int max_count = 0;
    for (int i = 0; i < 256; i++) {
        if (frequency[i] > max_count) {
            max_count = frequency[i];
            most_frequent = (char)i;
        }
    }

    free(labels);
    free(distances);
    free(frequency);

    return most_frequent;
}

int main() {
    int n_groups = parse_number_of_groups();
    Group *groups = (Group *)malloc(sizeof(Group) * n_groups);

    for (int i = 0; i < n_groups; i++) {
        groups[i] = parse_next_group();
    }

    int k = parse_k();
    Point to_evaluate = parse_point();

    printf("%c\n", knn(n_groups, groups, k, to_evaluate));

    for (int i = 0; i < n_groups; i++) free(groups[i].points);
    free(groups);

    return 0;
}
