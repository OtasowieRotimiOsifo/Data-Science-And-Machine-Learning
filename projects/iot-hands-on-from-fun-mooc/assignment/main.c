#include <stdio.h>

#include "thread.h"
#include "xtimer.h"

/* Add lps331ap related include here */
#include "lpsxxx.h"
#include "lpsxxx_params.h"

/* Add lsm303dlhc related include here */
#include "lsm303dlhc.h"
#include "lsm303dlhc_params.h"

/* Add isl29020 related include here */
#include "isl29020.h"
#include "isl29020_params.h"

/* Add l3g4200d related include here */
#include "l3g4200d.h"
#include "l3g4200d_params.h"

/* Declare the lps331ap device variable here */
static lpsxxx_t lpsxxx;

/* Declare the lsm303dlhc device variable here */
static lsm303dlhc_t lsm303dlhc;

/* Declare the isl29020 device variable here */
static isl29020_t isl29020;

/* Declare the l3g4200d device variable here */
static l3g4200d_t l3g4200d;

static char stack[THREAD_STACKSIZE_MAIN];
static char isl29020_stack[THREAD_STACKSIZE_MAIN];
static char l3g4200d_stack[THREAD_STACKSIZE_MAIN];

static void *l3g4200d_handler(void *arg)
{
(void)arg;

/* Add the isl29020 sensor polling endless loop here */
while (1) {
l3g4200d_data_t acc_data;
l3g4200d_read(&l3g4200d, &acc_data);

printf("Gyro data [dps] - X: %6i Y: %6i Z: %6i\n",
acc_data.acc_x, acc_data.acc_y, acc_data.acc_z);

xtimer_usleep(250 * US_PER_MS);
}
return 0;
}

static void *isl29020_handler(void *arg)
{
(void)arg;

/* Add the isl29020 sensor polling endless loop here */
while (1) {
int lux_value = isl29020_read(&isl29020);
printf("The Ligth Sensor Value is [lux]: %d \n", lux_value);

xtimer_usleep(250 * US_PER_MS);
}
return 0;
}

static void *lsm303dlhc_handler(void *arg)
{
(void)arg;

/* Add the lsm303dlhc sensor polling endless loop here */
while (1) {
lsm303dlhc_3d_data_t mag_value;
lsm303dlhc_3d_data_t acc_value;
lsm303dlhc_read_acc(&lsm303dlhc, &acc_value);
printf("Accelerometer x: %i y: %i z: %i\n",
acc_value.x_axis, acc_value.y_axis, acc_value.z_axis);
lsm303dlhc_read_mag(&lsm303dlhc, &mag_value);
printf("Magnetometer x: %i y: %i z: %i\n",
mag_value.x_axis, mag_value.y_axis, mag_value.z_axis);
xtimer_usleep(500 * US_PER_MS);
}
return 0;
}

int main(void)
{
/* Initialize the lps331ap sensor here */
lpsxxx_init(&lpsxxx, &lpsxxx_params[0]);

/* Initialize the lsm303dlhc sensor here */
lsm303dlhc_init(&lsm303dlhc, lsm303dlhc_params);


/* Initialize the isl29020 sensor here */
if(isl29020_init(&isl29020, &isl29020_params[0]) == 0) {
puts("Initializing the light sensor was ok\n");

} else {
puts("Initialization of light sensor: visl29020 failed");
return 1;
}

if (l3g4200d_init(&l3g4200d, &l3g4200d_params[0]) == 0) {
puts("Initialization of gyroscope was ok \n");
}
else {
puts("[Failed]");
return 1;
}

thread_create(stack, sizeof(stack), THREAD_PRIORITY_MAIN - 2,
0, lsm303dlhc_handler, NULL, "lsm303dlhc");

thread_create(isl29020_stack, sizeof(isl29020_stack), THREAD_PRIORITY_MAIN - 1,
0, isl29020_handler, NULL, "isl29020");

thread_create(l3g4200d_stack, sizeof(l3g4200d_stack), THREAD_PRIORITY_MAIN - 3,
0, l3g4200d_handler, NULL, "l3g4200d");

/* Add the lps331ap sensor polling endless loop here */
while (1) {
uint16_t pres = 0;
int16_t temp = 0;
lpsxxx_read_temp(&lpsxxx, &temp);
lpsxxx_read_pres(&lpsxxx, &pres);
printf("Pressure: %uhPa, Temperature: %u.%u°C\n",
pres, (temp / 100), (temp % 100));
xtimer_sleep(2);
}

return 0;
}
