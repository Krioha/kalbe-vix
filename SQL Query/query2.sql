select case
	WHEN gender  = 0 THEN 'Wanita'
	WHEN gender  = 1 THEN 'Pria'
end as "Jenis Kelamin" ,avg(age) as "Rata Rata usia"
from customer
group by gender