
import json
import os
import time
import random
import heapq
import statistics
import importlib


# ==========================================
# 1. LOAD PROJECT DATA
# ==========================================

def load_data():

    with open("data/parcels.json", "r") as file:
        parcels_data = json.load(file)

    with open("data/graph.json", "r") as file:
        graph_data = json.load(file)

    if isinstance(parcels_data, dict):
        parcels = parcels_data.get("parcels", [])
    else:
        parcels = parcels_data

    graph = normalize_graph(graph_data)

    return parcels, graph


def normalize_graph(data):

    graph = {}

    # Format 1: Nodes and edges
    if isinstance(data, dict) and "edges" in data:

        nodes = data.get("nodes", [])

        for node in nodes:
            name = node["id"] if isinstance(node, dict) else node
            graph[name] = []

        for edge in data["edges"]:
            source = edge.get("from", edge.get("source"))
            destination = edge.get("to", edge.get("target"))
            weight = edge.get("weight", 1)

            graph.setdefault(source, [])
            graph.setdefault(destination, [])

            graph[source].append((destination, weight))
            graph[destination].append((source, weight))

    # Format 2: Adjacency list
    else:

        for node, neighbors in data.items():

            graph[node] = []

            if isinstance(neighbors, dict):
                neighbors = list(neighbors.items())

            for neighbor, weight in neighbors:
                graph[node].append((neighbor, weight))

    return graph


# ==========================================
# 2. MERGE SORT
# ==========================================

def merge_sort(parcels):

    if len(parcels) <= 1:
        return parcels

    middle = len(parcels) // 2

    left = merge_sort(parcels[:middle])
    right = merge_sort(parcels[middle:])

    return merge(left, right)


def merge(left, right):

    result = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):

        key_left = (
            -left[i]["priority"],
            left[i]["deadline_hr"],
            str(left[i]["id"])
        )

        key_right = (
            -right[j]["priority"],
            right[j]["deadline_hr"],
            str(right[j]["id"])
        )

        if key_left <= key_right:
            result.append(left[i])
            i += 1

        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


# ==========================================
# 3. 0/1 KNAPSACK
# ==========================================

def knapsack(parcels, capacity):

    scale = 1000
    capacity_units = int(round(capacity * scale))

    n = len(parcels)

    dp = [
        [0] * (capacity_units + 1)
        for _ in range(n + 1)
    ]

    for i in range(1, n + 1):

        weight = int(round(parcels[i - 1]["weight"] * scale))
        value = int(parcels[i - 1]["value"])

        for w in range(capacity_units + 1):

            dp[i][w] = dp[i - 1][w]

            if weight <= w:

                dp[i][w] = max(
                    dp[i][w],
                    dp[i - 1][w - weight] + value
                )

    selected = []
    remaining = capacity_units

    for i in range(n, 0, -1):

        if dp[i][remaining] != dp[i - 1][remaining]:

            selected.append(parcels[i - 1])

            remaining -= int(
                round(parcels[i - 1]["weight"] * scale)
            )

    selected.reverse()

    return selected, dp[n][capacity_units]


# ==========================================
# 4. DIJKSTRA'S SHORTEST PATH
# ==========================================

def dijkstra(graph, start):

    distances = {
        node: float("inf")
        for node in graph
    }

    previous = {
        node: None
        for node in graph
    }

    if start not in graph:
        return distances, previous

    distances[start] = 0

    priority_queue = [(0, start)]

    while priority_queue:

        current_distance, current_node = heapq.heappop(
            priority_queue
        )

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:

            new_distance = current_distance + weight

            if new_distance < distances[neighbor]:

                distances[neighbor] = new_distance
                previous[neighbor] = current_node

                heapq.heappush(
                    priority_queue,
                    (new_distance, neighbor)
                )

    return distances, previous


def get_path(previous, destination):

    path = []
    current = destination

    while current is not None:
        path.append(current)
        current = previous[current]

    path.reverse()

    return path


# ==========================================
# 5. TSP - NEAREST NEIGHBOR
# ==========================================

def nearest_neighbor(graph, start):

    if start not in graph:
        return [], 0

    unvisited = set(graph.keys())
    unvisited.remove(start)

    route = [start]
    total_distance = 0
    current = start

    while unvisited:

        distances, _ = dijkstra(graph, current)

        reachable = [
            node for node in unvisited
            if distances[node] != float("inf")
        ]

        if not reachable:
            break

        next_node = min(
            reachable,
            key=lambda node: (distances[node], str(node))
        )

        total_distance += distances[next_node]

        route.append(next_node)
        unvisited.remove(next_node)

        current = next_node

    distances, _ = dijkstra(graph, current)

    if distances[start] != float("inf") and current != start:

        total_distance += distances[start]
        route.append(start)

    return route, total_distance


# ==========================================
# 6. CREATE TEST DATA FOR BENCHMARKING
# ==========================================

def create_test_parcels(n):

    parcels = []

    for i in range(n):

        parcels.append({
            "id": i,
            "dest": f"N{i}",
            "weight": random.randint(1, 5),
            "value": random.randint(10, 100),
            "priority": random.randint(1, 5),
            "deadline_hr": random.randint(1, 24)
        })

    return parcels


def create_test_graph(n):

    graph = {}

    for i in range(n):

        node = f"N{i}"
        graph[node] = []

    for i in range(n - 1):

        weight = random.randint(1, 10)

        graph[f"N{i}"].append(
            (f"N{i+1}", weight)
        )

        graph[f"N{i+1}"].append(
            (f"N{i}", weight)
        )

    return graph


# ==========================================
# 7. BENCHMARKING ALL FOUR ALGORITHMS
# ==========================================

def measure_time(function, repetitions=5):

    timings = []

    for _ in range(repetitions):

        start = time.perf_counter()

        function()

        end = time.perf_counter()

        timings.append(end - start)

    return statistics.median(timings)


def benchmark():

    sizes = [10, 25, 50, 100]

    rows = []

    print("\n========== PERFORMANCE BENCHMARK ==========")

    for n in sizes:

        print(f"\nTesting input size: {n}")

        parcels = create_test_parcels(n)
        graph = create_test_graph(n)

        # MERGE SORT
        merge_time = measure_time(
            lambda: merge_sort(parcels.copy())
        )

        # DIJKSTRA - ALL SOURCE NODES
        def run_all_dijkstra():

            for source in graph:
                dijkstra(graph, source)

        dijkstra_time = measure_time(
            run_all_dijkstra
        )

        # 0/1 KNAPSACK
        knapsack_time = measure_time(
            lambda: knapsack(parcels, 15)
        )

        # TSP
        tsp_time = measure_time(
            lambda: nearest_neighbor(graph, "N0")
        )

        rows.append((
            n,
            merge_time,
            dijkstra_time,
            knapsack_time,
            tsp_time
        ))

        print(f"Merge Sort: {merge_time:.8f} seconds")
        print(f"Dijkstra (all sources): {dijkstra_time:.8f} seconds")
        print(f"0/1 Knapsack: {knapsack_time:.8f} seconds")
        print(f"TSP: {tsp_time:.8f} seconds")

    return rows


# ==========================================
# 8. ONE COMBINED PERFORMANCE GRAPH
# ==========================================

def generate_chart(rows):

    try:

        plt = importlib.import_module(
            "matplotlib.pyplot"
        )

        os.makedirs("outputs", exist_ok=True)

        sizes = [row[0] for row in rows]

        merge_times = [
            max(row[1], 1e-9) for row in rows
        ]

        dijkstra_times = [
            max(row[2], 1e-9) for row in rows
        ]

        knapsack_times = [
            max(row[3], 1e-9) for row in rows
        ]

        tsp_times = [
            max(row[4], 1e-9) for row in rows
        ]

        plt.figure(figsize=(12, 7))

        plt.plot(
            sizes,
            merge_times,
            marker="o",
            linewidth=2,
            label="Merge Sort"
        )

        plt.plot(
            sizes,
            dijkstra_times,
            marker="s",
            linewidth=2,
            label="Dijkstra - All Sources"
        )

        plt.plot(
            sizes,
            knapsack_times,
            marker="^",
            linewidth=2,
            label="0/1 Knapsack"
        )

        plt.plot(
            sizes,
            tsp_times,
            marker="D",
            linewidth=2,
            label="TSP - Nearest Neighbor"
        )

        plt.yscale("log")

        plt.xticks(sizes)

        plt.xlabel(
            "Input Size (Number of Parcels / Nodes)",
            fontsize=11
        )

        plt.ylabel(
            "Execution Time (Seconds - Logarithmic Scale)",
            fontsize=11
        )

        plt.title(
            "Mini Courier Planner - Algorithm Performance Comparison",
            fontsize=14
        )

        plt.legend(loc="best")

        plt.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.6
        )

        plt.tight_layout()

        plt.savefig(
            "outputs/runtime_chart.png",
            dpi=300,
            bbox_inches="tight"
        )

        print("\nCombined graph saved successfully!")
        print("Location: outputs/runtime_chart.png")

        plt.show()

    except ImportError:

        print("\nMatplotlib is not available.")
        print("Please check your Python installation.")


# ==========================================
# 9. MAIN PROGRAM
# ==========================================

def main():

    print("\n")
    print("=" * 55)
    print("             MINI COURIER PLANNER")
    print("=" * 55)

    try:

        parcels, graph = load_data()

    except FileNotFoundError as error:

        print("\nFile not found:", error)
        print("Check your data folder.")
        return

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:

        print("\nProblem reading JSON data:", error)
        return

    if not parcels or not graph:

        print("\nParcel or graph data is empty.")
        return

    # MERGE SORT
    print("\n1. PARCELS SORTED BY PRIORITY")

    sorted_parcels = merge_sort(parcels.copy())

    for parcel in sorted_parcels:

        print(
            f"ID: {parcel['id']} | "
            f"Priority: {parcel['priority']} | "
            f"Deadline: {parcel['deadline_hr']} hours"
        )

    # KNAPSACK
    print("\n2. 0/1 KNAPSACK - SELECTED PARCELS")

    selected, total_value = knapsack(
        sorted_parcels,
        15
    )

    total_weight = sum(
        parcel["weight"] for parcel in selected
    )

    for parcel in selected:

        print(
            f"ID: {parcel['id']} | "
            f"Weight: {parcel['weight']} kg | "
            f"Value: {parcel['value']}"
        )

    print(f"Total weight: {total_weight} kg")
    print(f"Total value: {total_value}")

    # DIJKSTRA
    print("\n3. DIJKSTRA - SHORTEST PATHS FROM DEPOT")

    distances, previous = dijkstra(
        graph,
        "DEPOT"
    )

    for node in graph:

        if distances[node] == float("inf"):

            print(f"{node}: Unreachable")

        else:

            path = get_path(previous, node)

            print(
                f"{node}: Distance = {distances[node]} | "
                f"Path = {' -> '.join(path)}"
            )

    # ALL PAIRS SHORTEST PATHS
    print("\n4. DIJKSTRA - ALL PAIRS SHORTEST PATHS")

    for source in graph:

        distances, _ = dijkstra(graph, source)

        for destination in graph:

            print(
                f"{source} -> {destination}: "
                f"{distances[destination]}"
            )

    # TSP
    print("\n5. TSP - DELIVERY ROUTE")

    route, distance = nearest_neighbor(
        graph,
        "DEPOT"
    )

    print("Delivery route:", " -> ".join(route))
    print("Approximate route distance:", distance)

    # BENCHMARK
    print("\n6. PERFORMANCE COMPARISON")

    rows = benchmark()

    print("\n")
    print("=" * 75)
    print("             ALGORITHM EXECUTION TIME TABLE")
    print("=" * 75)

    print(
        f"{'Size':<10}"
        f"{'Merge Sort':<16}"
        f"{'Dijkstra':<16}"
        f"{'Knapsack':<16}"
        f"{'TSP':<16}"
    )

    for row in rows:

        print(
            f"{row[0]:<10}"
            f"{row[1]:<16.8f}"
            f"{row[2]:<16.8f}"
            f"{row[3]:<16.8f}"
            f"{row[4]:<16.8f}"
        )

    print("=" * 75)

    # ONE COMBINED GRAPH
    print("\n7. GENERATING COMBINED PERFORMANCE GRAPH")

    generate_chart(rows)

    print("\n")
    print("=" * 55)
    print("          PROGRAM COMPLETED SUCCESSFULLY")
    print("=" * 55)


if __name__ == "__main__":
    main()
