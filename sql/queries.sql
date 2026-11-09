-- Q1 WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS spend_over_500
FROM orders WHERE status='completed' AND order_amount > 500
GROUP BY customer_id HAVING SUM(order_amount) > 8000;

-- Q2 three-table join: revenue by city and category
SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue
FROM orders o JOIN customers c USING(customer_id)
JOIN products p USING(product_id)
WHERE o.status='completed'
GROUP BY c.city, p.category ORDER BY c.city, p.category;

-- Q3 products above average catalogue price
SELECT product_id, product_name, price_inr FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC;

-- Q4 customers without a completed order
SELECT c.customer_id FROM customers c LEFT JOIN orders o
ON c.customer_id=o.customer_id AND o.status='completed'
WHERE o.order_id IS NULL ORDER BY c.customer_id;

-- Q5 CTE: customers spending above average customer spend
WITH customer_spend AS (
 SELECT customer_id, SUM(order_amount) AS total_spend FROM orders
 WHERE status='completed' GROUP BY customer_id
), avg_spend AS (SELECT AVG(total_spend) AS avg_total FROM customer_spend)
SELECT cs.customer_id, cs.total_spend FROM customer_spend cs CROSS JOIN avg_spend a
WHERE cs.total_spend > a.avg_total ORDER BY cs.total_spend DESC;

-- Q6 ranking functions by city
WITH customer_spend AS (
 SELECT c.city, o.customer_id, SUM(o.order_amount) AS total_spend
 FROM orders o JOIN customers c USING(customer_id)
 WHERE o.status='completed' GROUP BY c.city,o.customer_id
)
SELECT city, customer_id, total_spend,
 RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS rank_in_city,
 DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_rank_in_city,
 ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM customer_spend ORDER BY city, rank_in_city, customer_id;

-- Q7 daily revenue and cumulative revenue
WITH daily AS (
 SELECT substr(order_ts,1,10) AS day, SUM(order_amount) AS revenue
 FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)
)
SELECT day, ROUND(revenue,2) AS revenue,
 ROUND(SUM(revenue) OVER(ORDER BY day),2) AS cumulative_revenue
FROM daily ORDER BY day;
