# Territory Tussle - CSoT Supply Chain Project

**CAIC Summer of Tech (CSoT)**  
**Week 1 Project — Supply Chain Domain**  
_By Economics and Finance Club × ANCC_

---

## 🏷️ Project Title: Optimizing Shared Logistics Zones

### 📘 Introduction

Supply chains are complex systems that coordinate the movement of goods, information, and resources across a network of suppliers, manufacturers, warehouses, and distributors. Problems in supply chains often revolve around:

- Minimizing costs  
- Maximizing efficiency  
- Ensuring timely delivery  
- Managing limited resources  

This project series introduces participants to the computational thinking needed to analyze and solve such real-world logistics problems. Starting from foundational geometric and graph-based challenges, the series builds toward more complex and computationally intensive tasks.

In the first two weeks, we focus on:

- Understanding spatial constraints  
- Overlapping service regions  
- Network-based representations of logistics flows  

Later weeks will cover classic combinatorial and packing problems, and eventually, computationally hard challenges. Participants will gain insights into not just the structure of supply chain issues, but also the **algorithms, heuristics, and trade-offs** used to tackle them in practice.

---

## 🧩 Problem Context

In the smart city of **Geometropolis**, two logistics giants — **HexaHaul Corp.** and **ConvexCarry Ltd.** — have proposed overlapping service areas. Each zone is a convex polygon defined by coordinates.

The central logistics authority wants you to compute the **exact area of overlap** between these regions.

This simulates real-world supply chain planning challenges like:

- Identifying overlapping distributor zones  
- Computing shared delivery regions  
- Planning buffer zones and minimizing redundancies

---

## 🧮 Problem Statement

You are given **T test cases**. Each case has:

- Two convex polygons (companies’ service areas)
- Your task: compute the **intersection area**, rounded to **4 decimal places**

### Assumptions

- All polygons are convex  
- Vertices listed in counterclockwise order  
- No vertex of one polygon lies on the edge of the other  
- Both polygons have non-zero area  

---

## 🧾 Input Format

T

N M

x1 y1 x2 y2 ... xN yN

x1’ y1’ x2’ y2’ ... xM’ yM’

Where:

- `T`: Number of test cases (1 ≤ T ≤ 10⁵)  
- `N, M`: Number of vertices in the 1st and 2nd polygon (3 ≤ N, M ≤ 5000)  
- Total polygon vertex pairs across all test cases ≤ 2.5 × 10⁷  
- Coordinates are integers: −10⁷ ≤ xi, yi ≤ 10⁷  

---

## 📤 Output Format

For each test case, print a **single line** with the **area of the intersection**, rounded to **four decimal places**.

---

## 📥 Sample Input

2

5 3

0 3 1 1 3 1 3 5 1 5

1 3 5 3 3 6

3 3

-1 -1 -2 -1 -1 -2

1 1 2 1 1 2


---

## 📤 Sample Output

2.6667

0.0000

---

## 📚 Further Resources to Learn About Supply Chain Problems

To better understand the real-world challenges and computational aspects of supply chains, explore the following resources:

### 🌐 Supply Chain Fundamentals
- [Coursera: Supply Chain Management Specialization by Rutgers](https://www.coursera.org/specializations/supply-chain-management) — Covers sourcing, logistics, planning, and strategy.
- [Khan Academy: Inventory and Supply Chain Management](https://www.khanacademy.org/economics-finance-domain/core-finance) — Good for economic context around inventory and flow.

### 📖 Articles & Case Studies
- [McKinsey: Supply Chain 4.0](https://www.mckinsey.com/business-functions/operations/our-insights/supply-chain-40--the-next-generation-digital-supply-chain) — How technology is reshaping logistics.
- [Harvard Business Review: Managing Uncertainty in Supply Chains](https://hbr.org/2003/10/the-supply-chain-has-no-clothes) — Insight into real-world issues and trade-offs.


